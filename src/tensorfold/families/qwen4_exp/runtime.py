"""Flash Next as the lane engine serves it: the fused decode plus the checkpoint's MTP head.

``FlashNext`` wraps the model (``model.py``) with what the lane engine's family rounds
(``engine.lane_family``) drive:

- ``exact_width``: the widest window (up to 16 rows) whose every row gets a one-row
  forward's bits, checked at load on this MLX and GPU, and ``window_costs``, each
  exact width's forward time; drafted rows are verified exactly;
- ``keep_rows``: roll every cache back to a prefix of a verified window;
- the MTP head: its cache absorbs each kept position (main-model streams and the
  sampled next token), then it chains ``drafts`` drafts from the last one. Drafts
  are sampled with the target's keyed sampler at their positions, so a draft is
  the target's own sample whenever the two distributions agree there.

Drafts change speed only: every emitted token is the target's sample.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.families.qwen4_exp.model import AttentionCache, LinearCache, _cache_state, _env_int


# -- exact drafting on the reference forward ------------------------------------------------------------------------
# A checkpoint the fused kernels cannot read (mixed precision, GGUF experts) decodes through the reference forward,
# whose multi-row calls sum rows in other orders than a one-row call: quantized and bf16 matmuls, attention over
# 3+ query rows and the indexer's float32 scores (measured with MLX 0.32.1 on an M4 Pro). TF_FLASH_REF_DRAFT
# verifies drafted windows there anyway, every row with the bits its one-row
# step gets:
#   pad    one-row steps and windows alike go through ``row_exact("pad")``: MLX's multi-row quantized matmul gives
#          a row the same bits at 2..12 rows and its bf16 matmul at 2..4 rows, so a one-row call runs as two
#          (the row twice) and a window's bf16 matmuls in pieces of 2..4 rows; attention and the indexer go one
#          query row at a time. Weights are read once a window. One-row steps then differ from the flag-off
#          build at the rounding level (M=2 kernels instead of M=1); drafted and undrafted output are identical.
#   batch  windows only, through ``row_exact("batch")``: quantized and bf16 matmuls as one vector product per row
#          (gather_qmm/gather_mm with a zero index, or one call per row), so a row gets the flag-off one-row
#          step's bits, at the cost of reading weights once per row.
#   chain  the window as one-row forwards queued back to back (exact by construction), each row's recurrent
#          state kept for the rollback.
#   off    (default) one token a round, as before.
# The rest of the forward (norms, softmax, convs, the DeltaNet recurrence, expert gathers under 64 routed rows) is
# row-invariant as it is. pad/batch roll a partly kept window back and run its kept rows again (same bits).
# TF_FLASH_REF_DRAFT_WIDTH: the widest window (default 8; 6 for batch, whose rows cost ~half a step each; at most
# 16; checked exact up to 12 on a 12-layer slice, M4 Pro). Expert gathers
# change kernels at 64 routed rows (10 a row), so wider windows take the MoE block in pieces of up to 6 rows.
# Every width is checked bit for bit at load, and the widest exact one is used.
_REF_MODES = ("off", "pad", "batch", "chain")


class _RowMasks:
    """Per-row attention masks from the row-wise indexer (None: that row is causal)."""

    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows


def _lead_rows(x: mx.array) -> int:
    n = 1
    for s in x.shape[:-1]:
        n *= int(s)
    return n


_ORIGINAL: dict[str, Any] = {}


def _qlinear_rows(self: Any, x: mx.array) -> mx.array:
    """A quantized linear as a vector product per row: gather_qmm with a zero index takes the one-row kernel's
    path for every row (checked: 6- and 8-bit, 640..12288 outputs, 2..16 rows)."""

    rows = _lead_rows(x)
    if rows <= 1:
        return _ORIGINAL["qlinear"](self, x)
    k = x.shape[-1]
    biases = self.get("biases")
    extra = {} if getattr(self, "mode", "affine") == "affine" else {"mode": self.mode}
    y = mx.gather_qmm(x.reshape(rows, 1, 1, k), self["weight"][None], self["scales"][None],
                      None if biases is None else biases[None], rhs_indices=mx.zeros((rows, 1), dtype=mx.uint32),
                      transpose=True, group_size=self.group_size, bits=self.bits, **extra)
    y = y.reshape(*x.shape[:-1], y.shape[-1])
    if "bias" in self:
        y = y + self["bias"]
    return y


def _linear_rows(self: Any, x: mx.array) -> mx.array:
    """A bf16 linear per row: gather_mm with a zero index for wide outputs (checked 2560/320 -> 2560..10240), one
    one-row call per row otherwise (its 10240 -> 320 differs from a one-row call; small outputs are not tested
    with enough values to trust). TF_FLASH_REF_LINEAR=split|gather forces one form."""

    rows = _lead_rows(x)
    if rows <= 1:
        return _ORIGINAL["linear"](self, x)
    w = self["weight"]
    n, k = int(w.shape[0]), int(w.shape[1])
    form = os.environ.get("TF_FLASH_REF_LINEAR", "auto")
    if "bias" not in self and (form == "gather" or (form == "auto" and n >= 2048 and k <= 4096)):
        y = mx.gather_mm(x.reshape(rows, 1, 1, k), w.T[None], rhs_indices=mx.zeros((rows, 1), dtype=mx.uint32))
        return y.reshape(*x.shape[:-1], n)
    flat = x.reshape(rows, 1, k)
    out = mx.concatenate([_ORIGINAL["linear"](self, flat[i:i + 1]) for i in range(rows)], axis=0)
    return out.reshape(*x.shape[:-1], n)


_PAD_QROWS = 12          # quantized matmul: rows 2..12 take one kernel, each row summed alike (16 does not)
_PAD_LROWS = 4           # bf16 matmul: rows 2..4 (6 does not)


def _padded(call: Any, module: Any, x: mx.array, most: int) -> mx.array:
    """``call(module, x)`` with every row summed as in a 2..``most``-row call: one row runs twice, longer calls in
    pieces of 2..most rows; calls past 16 rows (prompts) as they are."""

    rows = _lead_rows(x)
    if rows > 16 or (2 <= rows <= most):
        return call(module, x)
    k = x.shape[-1]
    flat = x.reshape(1, rows, k)
    if rows == 1:
        y = call(module, mx.concatenate([flat, flat], axis=1))[:, :1]
    else:
        pieces = -(-rows // most)
        sizes = [rows // pieces + (1 if i < rows % pieces else 0) for i in range(pieces)]
        outs, at = [], 0
        for size in sizes:
            outs.append(call(module, flat[:, at:at + size]))
            at += size
        y = mx.concatenate(outs, axis=1)
    return y.reshape(*x.shape[:-1], y.shape[-1])


def _qlinear_pad(self: Any, x: mx.array) -> mx.array:
    return _padded(_ORIGINAL["qlinear"], self, x, _PAD_QROWS)


# mlx-kquant's KQuantLinear (a GGUF trunk, gguf_trunk.py): one row takes the qmv route, 2..16 rows a split-K tile that
# each codec enters at its own row count unless KQ_QMM_SPLITK=1 forces it from 2 rows (read once per process). With
# it forced, a row gets the same bits in every call of 3..12 rows for all six trunk codecs (q4_k and q8_0 still take
# qmv at 2 rows; q5_k, q6_k, iq4_xs, iq4_nl agree from 2; 13..16 rows differ), measured on every codec/shape of the
# Coder trunk, MLX 0.32.1 / mlx-kquant 0.4.15, M4 Pro. So pad mode runs a kquant call as at least 3 rows
# (the last row repeated) and at most 12.
_KQ_MIN_ROWS, _KQ_MAX_ROWS = 3, 12


def _kqlinear_pad(self: Any, x: mx.array, lora: Any = None) -> mx.array:
    rows = _lead_rows(x)
    call = _ORIGINAL["kqlinear"]
    if lora is not None or rows > 16 or _KQ_MIN_ROWS <= rows <= _KQ_MAX_ROWS:
        return call(self, x, lora)
    k = x.shape[-1]
    flat = x.reshape(1, rows, k)
    if rows < _KQ_MIN_ROWS:
        filler = mx.repeat(flat[:, -1:], _KQ_MIN_ROWS - rows, axis=1)
        y = call(self, mx.concatenate([flat, filler], axis=1))[:, :rows]
    else:                                              # 13..16 rows: two pieces of 6..8
        half = rows // 2
        y = mx.concatenate([call(self, flat[:, :half]), call(self, flat[:, half:])], axis=1)
    return y.reshape(*x.shape[:-1], y.shape[-1])


def _kqlinear_rows(self: Any, x: mx.array, lora: Any = None) -> mx.array:
    """Batch mode: one one-row call per row, so every row takes the flag-off qmv route."""

    rows = _lead_rows(x)
    if lora is not None or rows <= 1 or rows > 16:
        return _ORIGINAL["kqlinear"](self, x, lora)
    flat = x.reshape(rows, 1, x.shape[-1])
    out = mx.concatenate([_ORIGINAL["kqlinear"](self, flat[i:i + 1]) for i in range(rows)], axis=0)
    return out.reshape(*x.shape[:-1], out.shape[-1])


def _kquant_linear() -> Any:
    try:
        from mlx_kquant.nn import KQuantLinear
    except ImportError:
        return None
    return KQuantLinear


def _linear_pad(self: Any, x: mx.array) -> mx.array:
    return _padded(_ORIGINAL["linear"], self, x, _PAD_LROWS)


_MOE_ROWS = 6           # the expert gathers keep one kernel per row below 64 routed rows (10 a row)


def _moe_rows(self: Any, x: mx.array) -> mx.array:
    """A window wider than 6 rows through the MoE block in pieces of at most 6 rows (every op in it is row-wise)."""

    rows = _lead_rows(x)
    moe = _ORIGINAL["moe"]
    if rows <= _MOE_ROWS or rows > 16:
        return moe(self, x)
    flat = x.reshape(1, rows, x.shape[-1])
    pieces = -(-rows // _MOE_ROWS)
    sizes = [rows // pieces + (1 if i < rows % pieces else 0) for i in range(pieces)]
    outs, at = [], 0
    for size in sizes:
        outs.append(moe(self, flat[:, at:at + size]))
        at += size
    y = mx.concatenate(outs, axis=1)
    return y.reshape(*x.shape[:-1], y.shape[-1])


def _sdpa_rows(queries: mx.array, keys: mx.array, values: mx.array, *, scale: float, mask: Any = None,
               **kwargs: Any) -> mx.array:
    """Attention one query row at a time, keys cut at the row's causal end, with the row's own mask: what a
    one-row decode step calls."""

    rows = int(queries.shape[2])
    sdpa = _ORIGINAL["sdpa"]
    if rows <= 1:
        return sdpa(queries, keys, values, scale=scale, mask=mask, **kwargs)
    past = int(keys.shape[2]) - rows
    outs = []
    for i in range(rows):
        end = past + i + 1
        if isinstance(mask, _RowMasks):
            m = mask.rows[i]
        elif isinstance(mask, mx.array):
            m = mask[..., i:i + 1, :end]
        else:
            m = None
        outs.append(sdpa(queries[:, :, i:i + 1], keys[:, :, :end], values[:, :, :end], scale=scale, mask=m,
                         **kwargs))
    return mx.concatenate(outs, axis=2)


def _select_rows(self: Any, query: mx.array, raw: mx.array, cache: Any, past: int) -> Any:
    """The indexer's block choice one row at a time (pooling blocks as each row completes them), as decode."""

    select = _ORIGINAL["select"]
    length = int(query.shape[1])
    if length <= 1:
        return select(self, query, raw, cache, past)
    masks = [select(self, query[:, i:i + 1], raw[:, :past + i + 1], cache, past + i) for i in range(length)]
    if all(m is None for m in masks):
        return None
    return _RowMasks(masks)


@contextlib.contextmanager
def row_exact(mode: str = "batch"):
    """Inside: the reference forward's (and head's) calls give each row the bits ``mode`` promises (above). Not
    re-entrant across threads: the lane engine runs one forward at a time."""

    import mlx.nn as nn

    from tensorfold.families.qwen4_exp import model as q4

    if _ORIGINAL.get("active"):
        yield
        return
    kq_linear = _kquant_linear()
    _ORIGINAL.update(qlinear=nn.QuantizedLinear.__call__, linear=nn.Linear.__call__,
                     sdpa=mx.fast.scaled_dot_product_attention, select=q4.Indexer.select,
                     moe=q4.SparseMoE.__call__, active=True)
    pad = mode == "pad"
    nn.QuantizedLinear.__call__ = _qlinear_pad if pad else _qlinear_rows
    nn.Linear.__call__ = _linear_pad if pad else _linear_rows
    if kq_linear is not None:
        _ORIGINAL["kqlinear"] = kq_linear.__call__
        kq_linear.__call__ = _kqlinear_pad if pad else _kqlinear_rows
    mx.fast.scaled_dot_product_attention = _sdpa_rows
    q4.Indexer.select = _select_rows
    q4.SparseMoE.__call__ = _moe_rows
    try:
        yield
    finally:
        nn.QuantizedLinear.__call__ = _ORIGINAL["qlinear"]
        nn.Linear.__call__ = _ORIGINAL["linear"]
        if kq_linear is not None:
            kq_linear.__call__ = _ORIGINAL["kqlinear"]
        mx.fast.scaled_dot_product_attention = _ORIGINAL["sdpa"]
        q4.Indexer.select = _ORIGINAL["select"]
        q4.SparseMoE.__call__ = _ORIGINAL["moe"]
        _ORIGINAL["active"] = False


def _deltanet_recording(self: Any, x: mx.array, cache: LinearCache) -> mx.array:
    """GatedDeltaNet.__call__ (model.py) that also keeps a drafted window's recurrence inputs on the cache
    (``window_inputs``), so a rollback can replay the kept rows' recurrence instead of re-running the model."""

    import mlx.nn as nn

    from tensorfold.families.qwen4_exp.model import _own_tail, gated_delta_update

    batch, length, _ = x.shape
    qkv = self.in_proj_qkv(x)
    z = self.in_proj_z(x).reshape(batch, length, self.nv, self.dv)
    b = self.in_proj_b(x)
    a = self.in_proj_a(x)
    tail = cache.conv if cache.conv is not None else mx.zeros((batch, self.kernel - 1, self.conv_dim), x.dtype)
    conv_in = mx.concatenate([tail, qkv], axis=1)
    cache.conv = _own_tail(conv_in, self.kernel - 1)
    conv = nn.silu(self.conv1d(conv_in))
    q, k, v = mx.split(conv, [self.key_dim, 2 * self.key_dim], axis=-1)
    q = q.reshape(batch, length, self.nk, self.dk)
    k = k.reshape(batch, length, self.nk, self.dk)
    v = v.reshape(batch, length, self.nv, self.dv)
    q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6) * (self.dk ** -0.5)
    k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
    if self.v_tiled and self.nv > self.nk:
        q = mx.tile(q, (1, 1, self.nv // self.nk, 1))
        k = mx.tile(k, (1, 1, self.nv // self.nk, 1))
    cache.window_inputs = (conv_in, q, k, v, a, b)
    out, cache.ssm = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, cache.ssm)
    cache.offset += length
    out = self.norm(out, z)
    return self.out_proj(out.reshape(batch, length, -1))


@contextlib.contextmanager
def _recording_windows():
    from tensorfold.families.qwen4_exp import model as q4

    original = q4.GatedDeltaNet.__call__
    q4.GatedDeltaNet.__call__ = _deltanet_recording
    try:
        yield
    finally:
        q4.GatedDeltaNet.__call__ = original


def _replay_ok() -> bool:
    """TF_FLASH_REF_REPLAY=0: a partly kept window re-runs its kept rows through the model (the old rollback)."""

    return os.environ.get("TF_FLASH_REF_REPLAY", "1").strip() != "0"


def _linear_state(cache: LinearCache) -> tuple[Any, ...]:
    return cache.conv, cache.ssm, cache.ple_conv, cache.history, cache.offset


def _restore_linear(cache: LinearCache, state: tuple[Any, ...]) -> None:
    cache.conv, cache.ssm, cache.ple_conv, cache.history, cache.offset = state
    cache.ple_rollback = None


class MTPCache(AttentionCache):
    """The MTP head's attention cache; ``drafted``: how many of its last entries are chained drafts (trimmed before
    the next kept rows are absorbed)."""

    drafted = 0


class FlashNext:
    """Flash Next with the backbone and head apart, the fused decode, and MTP drafting."""

    fused_rows = 16
    lane_family = True
    # tokens drawn by gpu_sampling (the exact keyed rule in one kernel): the host sampler's argpartition of
    # 248k logits and readback cost ~2 ms a draw on the M3 Ultra, twice a drafted round (2026-09-25)
    gpu_sampling = True

    def __init__(self, model: Any, head: Any | None = None, *, drafts: int = 1) -> None:
        self.model = model
        self.args = model.args
        self.fused = model.__dict__.get("fused")
        self.layer_count = len(model.layers)
        self.mtp = None
        self.drafts = int(drafts)
        self.ref_draft = "off"
        self._ref_round: Any = None
        mode = os.environ.get("TF_FLASH_REF_DRAFT", "off").strip().lower() or "off"
        if mode not in _REF_MODES:
            raise ValueError(f"TF_FLASH_REF_DRAFT={mode!r}: one of {', '.join(_REF_MODES)}")
        if self.fused is None and mode != "off":
            # a GGUF trunk (KQuantLinear) is row-exact in pad mode only with kquant's split-K route forced from 2 rows
            # (see _kqlinear_pad); kquant reads this once per process, so launchers set it too; the check below decides
            os.environ.setdefault("KQ_QMM_SPLITK", "1")
            self.ref_draft = mode
            widest = max(1, min(16, _env_int("TF_FLASH_REF_DRAFT_WIDTH", 6 if mode == "batch" else 8)))
            self.exact_width = widest                    # the check's windows take the drafted path
            self.exact_width, self.window_costs = self.check_ref_windows(widest)
            print(f"[flash-next] reference-forward drafting ({mode}): exact windows up to {self.exact_width} rows, "
                  f"forward ms by width {self.window_costs}", flush=True)
            if self.exact_width < 2:
                self.ref_draft = "off"
                print("[flash-next] a drafted window did not reproduce one-row steps: no drafts", flush=True)
        elif self.fused is not None:
            self.exact_width, self.window_costs = self.check_windows()
        else:
            self.exact_width, self.window_costs = 1, {}
        self.multi_row_exact = self.exact_width >= 2
        if self.fused is not None and not self.multi_row_exact:
            print("[flash-next] a multi-row forward does not reproduce serial steps on this MLX/GPU: no drafts",
                  flush=True)
        if self.multi_row_exact and head is not None and self.drafts > 0 and self.fused is None:
            # a GGUF head (gguf_head.py) on the reference forward: its layer runs the reference modules at every row
            # count and drafts are drawn over the whole vocabulary (the cut head needs a 4-bit g32 lm_head); drafts
            # only change speed, so the head need not be row-exact
            self.mtp = head
            self.mtp_fused = None
            # drafts over the fixed id list (draft_head.py) from the rows of the GGUF vocabulary head: K-quant rows
            # are independent bytes, so the cut head's logits equal the full head's at those ids (167 of 521 MB
            # read a draft step for a Q6_K head; TF_FLASH_DRAFT_VOCAB=0: the whole vocabulary)
            self._draft_ids = self._draft_head = None
            lm = model.lm_head
            if os.environ.get("TF_FLASH_DRAFT_VOCAB", "1") != "0" and getattr(lm, "kquant_type", None):
                from mlx_kquant.nn import KQuantLinear

                from tensorfold.families.qwen4_exp.draft_head import draft_ids

                ids = draft_ids()
                cut = KQuantLinear(int(self.args.hidden_size), len(ids), False, lm.kquant_type)
                cut.load_weights([("weight", mx.contiguous(lm["weight"][mx.array(ids)])), ("scales", lm["scales"])])
                mx.eval(cut.parameters())
                self._draft_ids, self._draft_head = mx.array(ids), cut
            self.queued_chains = os.environ.get("TF_FLASH_QUEUED", "1") != "0"
            self.mtp_step_ms = self._time_mtp_step()
            print(f"[flash-next] MTP head on the reference forward ({getattr(head, 'gguf_source', {})}), "
                  f"{self.drafts} drafts at most, step {self.mtp_step_ms} ms", flush=True)
        elif self.multi_row_exact and head is not None and self.drafts > 0:
            self.mtp = head
            # the head's decoder layer and mixer through the same fused kernels as the model's layers
            from types import SimpleNamespace

            from tensorfold.families.qwen4_exp.decode import FusedDecode

            shell = SimpleNamespace(args=self._head_config(), layers=head.layers,
                                    model=SimpleNamespace(hyper_connection_mixer=head.hyper_connection_mixer,
                                                          embed_tokens=model.model.embed_tokens))
            self.mtp_fused = FusedDecode(shell)
            self._mtp_scales = [1.0 + head.pre_fc_norm_embedding.weight.astype(mx.float32),
                                1.0 + head.pre_fc_norm_hidden.weight.astype(mx.float32)]
            mx.eval(*self._mtp_scales)

            # the head scores a fixed list of 79,591 ids (TF_FLASH_DRAFT_VOCAB=0: the whole vocabulary), and a
            # round's chained drafts stay on the GPU until the next round reads them (TF_FLASH_QUEUED=0: each read)
            self._draft_ids = self._draft_head = None
            if os.environ.get("TF_FLASH_DRAFT_VOCAB", "1") != "0":
                from tensorfold.families.qwen4_exp.draft_head import cut_head, draft_ids

                ids = draft_ids()
                self._draft_ids = mx.array(ids)
                self._draft_head = cut_head(model.lm_head, ids)
            self.queued_chains = os.environ.get("TF_FLASH_QUEUED", "1") != "0"
            self.mtp_step_ms = self._time_mtp_step()

    queued_chains = False

    mtp_step_ms = 0.0

    def _time_mtp_step(self) -> float:
        """One chained draft step as ``settle`` takes it (the head's layer, the vocabulary head, a draw read back),
        ms, fastest of 6: the engine's depth rule adds it per draft before measured rounds replace the estimate."""

        import time

        cache = MTPCache()
        if self.fused is not None:
            streams = self.fused.last_streams[-1:]
        else:
            streams = mx.zeros((1, self.mtp.streams * int(self.args.hidden_size)), dtype=mx.bfloat16)
        mixed, out = self._mtp_step([3001], streams, cache)
        mx.eval(mixed, out)
        best = float("inf")
        for i in range(6):
            started = time.perf_counter()
            mixed, out = self._mtp_step([3002 + i], out, cache)
            self._draft_draw(mixed, None, [100 + i]).item()
            best = min(best, (time.perf_counter() - started) * 1e3)
        return round(best, 3)

    # -- the serial engine's model interface ----------------------------------------
    @property
    def layers(self) -> list[Any]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        caches = self.model.make_cache()
        if self.mtp is not None:
            caches.append(MTPCache())                         # last: the model's layers never reach it
        return caches

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        """A stored or copied cache without an MTP entry gets an empty one (drafts then see less context). With
        TF_KV_PAGED=1 its plain attention layers become ``paged_kv.HybridAttentionCache`` (``paged_kv.adopt``)."""

        if self.mtp is not None and len(cache) == self.layer_count:
            cache.append(MTPCache())
        from tensorfold.families.qwen4_exp import paged_kv

        return paged_kv.adopt(cache) if paged_kv.enabled() else cache

    def attach_prompt_cache(self, *, evict: Any, context_window: int) -> None:
        """Called by the server once its prompt cache exists. TF_KV_PAGED=1: the page pools take their size from
        the context window and evict the cache's least recently used entries when full (``paged_kv.attach``)."""

        from tensorfold.families.qwen4_exp import paged_kv

        if paged_kv.enabled():
            paged_kv.attach(evict, context_window)

    # Prompts through the decode kernels, ``fused_rows`` rows at a time: each prompt row then gets a one-row step's
    # bits whatever the chunking, so a prompt resumed from any cached prefix (a checkpoint, a finished reply's
    # cache) gives what the same prompt fed fresh gives. MLX's prefill kernels sum a row differently by chunk: a
    # second chat turn resumed from the first differed from the same turn fed fresh (thinking on, M3 Ultra,
    # 2026-09-26). False: MLX's prefill (faster, not exact across chunkings).
    exact_prefill = True

    def hidden(self, inputs: Any, cache: list[Any]) -> mx.array:
        tokens = np.asarray(inputs, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        rows = tokens.shape[1]
        self._ref_round = None
        if self.ref_draft != "off" and rows <= self.exact_width and tokens.shape[0] == 1:
            if rows > 1:
                return self._ref_window(tokens, cache)
            if self.ref_draft == "pad":               # one-row steps through the same kernels as the windows
                with row_exact("pad"):
                    out = self.model.hidden(tokens, cache[: self.layer_count])
                self._streams = self.model.__dict__["last_streams"]
                return out
        if self.fused is not None and rows > self.fused_rows and self.exact_prefill and tokens.shape[0] == 1:
            layers = cache[: self.layer_count]
            outs, streams = [], []
            for begin in range(0, rows, self.fused_rows):
                outs.append(self.fused(tokens[:, begin:begin + self.fused_rows], layers))
                streams.append(self.fused.last_streams)
                # one window in flight behind the one being built: the host builds a window in a few ms and the
                # GPU runs it in ~60, and every window's per-row DeltaNet states (1.8 GB) stay alive until it runs
                mx.async_eval(outs[-1], streams[-1])
                if len(outs) > 1:
                    mx.eval(outs[-2])
            self._streams = mx.concatenate(streams)
            return mx.concatenate(outs, axis=1)
        # TF_FLASH_PREFILL_ROWS=n (reference forward only): a prompt chunk longer than n rows runs as n-row calls,
        # each evaluated before the next: the forward's per-row buffers (MoE gathers, float32 hyper-connection
        # norms, attention scores) scale with the rows of a call. Not bit-identical to one call: MLX's matmuls
        # and the recurrences' chunking sum in another order for other row counts.
        step = _env_int("TF_FLASH_PREFILL_ROWS") if self.fused is None else 0
        if step > self.fused_rows and rows > step and tokens.shape[0] == 1:
            layers = cache[: self.layer_count]
            outs, streams = [], []
            for begin in range(0, rows, step):
                outs.append(self.model.hidden(tokens[:, begin:begin + step], layers))
                streams.append(self.model.__dict__["last_streams"])
                mx.eval(outs[-1], streams[-1], *[a for c in layers for a in _cache_state(c)])
            self._streams = mx.concatenate(streams)
            return mx.concatenate(outs, axis=1)
        out = self.model.hidden(tokens, cache[: self.layer_count])
        fused = self.fused is not None and rows <= self.fused_rows
        self._streams = self.fused.last_streams if fused else self.model.__dict__["last_streams"]
        return out

    def head(self, hidden: mx.array) -> mx.array:
        from tensorfold.families.qwen4_exp.decode import project

        head = self.model.lm_head
        # project()'s multi-row kernel reads 4-bit group-32 weights only; any other head takes MLX's own matmul
        if (getattr(head, "bits", None), getattr(head, "group_size", None)) != (4, 32):
            if self.ref_draft == "pad" or (self.ref_draft != "off" and _lead_rows(hidden) > 1):
                with row_exact("pad" if self.ref_draft == "pad" else "batch"):   # each row as its one-row step's
                    return head(hidden)
            return head(hidden)
        return project(hidden, head)

    def __call__(self, inputs: Any, cache: list[Any]) -> mx.array:
        return self.head(self.hidden(inputs, cache))

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        if self.fused is not None:
            self.fused.keep_rows(cache, rows, keep)
            return
        self._ref_keep(cache, rows, keep)

    # -- drafted windows on the reference forward (TF_FLASH_REF_DRAFT) ------------------------------------------
    def _ref_window(self, tokens: np.ndarray, cache: list[Any]) -> mx.array:
        """A drafted window [1, R] through the reference forward, each row with a one-row step's bits; remembers
        what ``keep_rows`` needs to roll it back."""

        layers = cache[: self.layer_count]
        rows = int(tokens.shape[1])
        before = [_linear_state(c) if isinstance(c, LinearCache) else None for c in layers]
        states = None
        if self.ref_draft == "chain":
            outs, states = [], []
            for r in range(rows):
                out = self.model.hidden(tokens[:, r:r + 1], layers)
                outs.append(out)
                states.append([_linear_state(c) if isinstance(c, LinearCache) else None for c in layers])
                mx.async_eval(out)                   # the GPU runs row r while the host builds row r + 1
            hidden = mx.concatenate(outs, axis=1)
        else:
            for c in layers:
                if isinstance(c, LinearCache):
                    c.window_inputs = None
            record = _recording_windows() if _replay_ok() else contextlib.nullcontext()
            with row_exact(self.ref_draft), record:
                hidden = self.model.hidden(tokens, layers)
        self._streams = self.model.__dict__["last_streams"]
        self._ref_round = (id(cache), tokens, before, states)
        return hidden

    def _ref_keep(self, cache: list[Any], rows: int, keep: int) -> None:
        """Keep the first ``keep`` rows of the last drafted window: attention caches trimmed; DeltaNet/PLE state as
        after row keep - 1 (chain: kept per row; batch: rolled back to before the window, then the kept rows run
        again, which gives them the same bits)."""

        held = self._ref_round
        self._ref_round = None
        if held is None or held[0] != id(cache) or int(held[1].shape[1]) != rows:
            raise RuntimeError("keep_rows: no drafted reference window of that size on this cache")
        _, tokens, before, states = held
        layers = cache[: self.layer_count]
        ratio = self.args.indexer_compress_ratio
        if states is not None:
            for c, state in zip(layers, states[keep - 1]):
                if state is None:
                    c.trim(rows - keep, ratio)
                else:
                    _restore_linear(c, state)
            return
        linear = [c for c, state in zip(layers, before) if state is not None]
        if _replay_ok() and all(getattr(c, "window_inputs", None) is not None for c in linear):
            # replay: attention caches drop the rejected rows; each DeltaNet takes its conv tail from the window's
            # conv input and replays the kept rows' recurrence one row at a time from the state before the window
            # (a one-row step's call, so the same bits); PLE layers keep the window's first rows (as decode.py)
            from tensorfold.families.qwen4_exp.model import gated_delta_update

            for layer, c, state in zip(self.model.layers, layers, before):
                if state is None:
                    c.trim(rows - keep, ratio)
                    continue
                net = layer.linear_attn
                conv_in, q, k, v, a, b = c.window_inputs
                c.window_inputs = None
                c.conv = mx.contiguous(conv_in[:, keep:keep + net.kernel - 1])
                ssm = state[1]
                for r in range(keep):
                    _, ssm = gated_delta_update(q[:, r:r + 1], k[:, r:r + 1], v[:, r:r + 1], a[:, r:r + 1],
                                                b[:, r:r + 1], net.A_log, net.dt_bias, ssm)
                c.ssm = ssm
                c.offset = state[4] + keep
                if "ple" in layer and c.ple_rollback is not None:
                    history, toks, ple_in = c.ple_rollback
                    ple = layer.ple
                    c.history = np.concatenate([history, toks[:, :keep]], axis=1)[:, -ple.ple_embedding.context:]
                    c.ple_conv = mx.contiguous(ple_in[:, keep:keep + ple.tail])
                    c.ple_rollback = None
            mx.async_eval(*[a for c in layers for a in _cache_state(c)])
            return
        for c, state in zip(layers, before):
            if state is None:
                c.trim(rows, ratio)
            else:
                _restore_linear(c, state)
        exact = row_exact(self.ref_draft) if keep > 1 or self.ref_draft == "pad" else contextlib.nullcontext()
        with exact:
            self.model.hidden(tokens[:, :keep], layers)
        mx.async_eval(*[a for c in layers for a in _cache_state(c)])

    def check_ref_windows(self, widest: int) -> tuple[int, dict[int, float]]:
        """``check_windows`` for the reference forward's drafted windows: the widest window (up to ``widest``)
        whose every narrower window gives each row a one-row step's logits bit for bit, after a 48-token prompt
        and again after a rollback (keep half the window, then the rest one row at a time); every exact width's
        forward time in ms (fastest of 3)."""

        import time

        from tensorfold.engine.lane_engine import LaneEngine

        copy = LaneEngine.copy_single_cache
        vocab = int(self.args.vocab_size)               # ids in range for any vocabulary (tests use tiny ones)
        prompt = np.array([[((37 * i + 11) % 50_000 + 1000) % vocab for i in range(48)]], dtype=np.int64)
        window = [(3001 + 17 * r) % vocab for r in range(widest)]
        base = self.model.make_cache()
        mx.eval(self.model.hidden(prompt, base))
        one = copy(base)
        serial = []
        for token in window:
            logits = self.head(self.hidden(np.array([[token]], dtype=np.int64), one))
            mx.eval(logits)
            serial.append(logits[0, -1])
        exact = 1
        for width in range(2, widest + 1):
            work = copy(base)
            logits = self.head(self.hidden(np.array([window[:width]], dtype=np.int64), work))
            mx.eval(logits)
            if not all(bool(mx.array_equal(logits[0, i], serial[i]).item()) for i in range(width)):
                break
            # roll back to half the window, then the rest of it one row at a time: the same logits again
            keep = max(1, width // 2)
            self.keep_rows(work, width, keep)
            ok = True
            for i in range(keep, width):
                again = self.head(self.hidden(np.array([[window[i]]], dtype=np.int64), work))
                mx.eval(again)
                ok = ok and bool(mx.array_equal(again[0, -1], serial[i]).item())
            if not ok:
                break
            exact = width
        costs: dict[int, float] = {}
        for width in range(1, exact + 1):
            best = float("inf")
            for _ in range(3):
                work = copy(base)
                started = time.perf_counter()
                mx.eval(self.head(self.hidden(np.array([window[:width]], dtype=np.int64), work)))
                best = min(best, (time.perf_counter() - started) * 1e3)
            costs[width] = round(best, 3)
        self._ref_round = None
        return exact, costs

    # -- drafting -------------------------------------------------------------------
    @property
    def last_streams(self) -> mx.array:
        """The last hidden() call's residual streams before the final mixer, [L, S*D]."""

        return self._streams

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any]) -> None:
        """The MTP cache takes the last hidden() call's first len(next_tokens) positions (prompt rows)."""

        tokens = [int(t) for t in np.asarray(next_tokens).reshape(-1)]
        self._absorb(self._streams[:len(tokens)], tokens, cache[-1])

    def _head_config(self) -> Any:
        from dataclasses import replace

        return replace(self.args, num_hidden_layers=1, layer_types=["sparse_attention"], ple_layer_ids=[])

    def _mtp_step(self, tokens: Any, streams: mx.array, mtp_cache: MTPCache) -> tuple[mx.array, mx.array]:
        """The MTP head on rows (next tokens, residual streams [n, S*D]): (mixed [1, n, D], its streams [n, S*D]).
        Prompt-long inputs take the reference modules; decode rows the fused kernels."""

        head = self.mtp
        rows, wide = streams.shape
        dims = wide // head.streams
        if self.__dict__.get("mtp_fused") is not None and rows <= self.fused_rows:
            # the prologue in a few kernels (was ~20 MLX ops a draft step): embedding rows, the two (1 + w) norms,
            # the two projections through ``project``
            from tensorfold.kernels.qwen.flash_next.v1 import kernels as K
            from tensorfold.families.qwen4_exp.decode import project

            eps = self.mtp_fused.eps
            emb = K.embed_rows(tokens, self.model.model.embed_tokens)                      # [n, D]
            e = project(K.rms_norm_rows(emb, self._mtp_scales[0], eps), head.fc_embedding)
            normed = K.rms_norm_rows(streams, self._mtp_scales[1], eps).reshape(rows * head.streams, dims)
            hs = project(normed, head.fc_hidden) if rows * head.streams <= 32 else head.fc_hidden(normed)
            x = (e[:, None, :] + hs.reshape(rows, head.streams, dims)).reshape(rows, wide)
            mixed = self.mtp_fused.run(x, None, [mtp_cache])
            return mixed, self.mtp_fused.last_streams
        ids = tokens.astype(mx.int32) if isinstance(tokens, mx.array) else mx.array(tokens, dtype=mx.int32)
        emb = self.model.model.embed_tokens(ids)                                            # [n, D]
        e = head.fc_embedding(head.pre_fc_norm_embedding(emb))
        hs = head.fc_hidden(head.pre_fc_norm_hidden(streams).reshape(rows, head.streams, dims))
        x = (e[:, None, :] + hs).reshape(rows, wide)
        x = head.layers[0](x[None], None, mtp_cache)
        return head.hyper_connection_mixer(x), x[0]

    def _draft_draw(self, mixed: mx.array, sampling: Any, positions: Any) -> mx.array:
        """Drafts (uint32 [n], lazy) from the head's mixed hidden states [1, n, D]: the target's keyed rule over the
        listed ids when the head is cut, else over the whole vocabulary."""

        from tensorfold.families.qwen4_exp.decode import project

        x = mixed.reshape(-1, mixed.shape[-1])
        if self._draft_head is not None:
            from tensorfold.families.qwen4_exp.draft_head import sample as draft_sample

            kq_head = getattr(self._draft_head, "kquant_type", None) is not None
            logits = self._draft_head(x) if kq_head else project(x, self._draft_head)
            return draft_sample(logits, self._draft_ids, sampling, positions)
        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        return gpu_sample(self.head(x[None]).reshape(x.shape[0], -1), sampling, positions)

    _draft_head: Any = None
    _draft_ids: Any = None

    def _absorb(self, streams: mx.array, tokens: list[int], mtp_cache: MTPCache) -> tuple[mx.array, mx.array]:
        if mtp_cache.drafted:
            mtp_cache.trim(mtp_cache.drafted, self.args.indexer_compress_ratio)
            mtp_cache.drafted = 0
        mixed, out = self._mtp_step(tokens, streams, mtp_cache)
        return mixed[:, -1:], out[-1:]

    def draft(self, cache: list[Any], streams: mx.array, tokens: list[int], position: int, sampling: Any,
              count: int | None = None) -> list[int]:
        """Absorb positions whose residual streams are ``streams`` [n, S*D] and whose next tokens are ``tokens``,
        then chain ``count`` (default ``drafts``) drafts for positions ``position``, ``position`` + 1, ..."""

        mtp_cache = cache[-1]
        mixed, out = self._absorb(streams, [int(t) for t in tokens], mtp_cache)
        drafts: list[int] = []
        count = self.drafts if count is None else int(count)
        for j in range(count):
            d = int(self._draft_draw(mixed, sampling, [position + j]).item())
            drafts.append(d)
            if j + 1 < count:
                mixed, out = self._mtp_step([d], out, mtp_cache)
                mtp_cache.drafted += 1
        return drafts

    def speculate(self, cache: list[Any], tokens: mx.array, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False) -> mx.array:
        """Before a verify round's tokens are read: the MTP head absorbs rows ``start`` .. of the last hidden()
        call (their streams; ``tokens`` [n], the tokens that follow them, still on the GPU) and draws each row's
        first draft, for positions ``position`` + 2 + i (``position``: row ``start``'s). One read then returns
        both; ``settle`` keeps the kept rows' part. Rows go through the MTP head exactly as ``draft`` would take
        the kept ones (the fused kernels give a row the same bits at any row count), so the drafts and the MTP
        cache are the same, a host round trip earlier. Returns the drafts [n] (lazy)."""

        mtp_cache = cache[-1]
        if mtp_cache.drafted:
            mtp_cache.trim(mtp_cache.drafted, self.args.indexer_compress_ratio)
            mtp_cache.drafted = 0
        tokens = tokens.reshape(-1)
        rows = int(tokens.shape[0])
        total = int(self._streams.shape[0])
        start = start + total if start < 0 else start
        mixed, out = self._mtp_step(tokens, self._streams[start:start + rows], mtp_cache)
        self._spec = (out, rows)
        if last_only:                      # the last row's draft only (every row still enters the head's cache)
            return self._draft_draw(mixed[:, -1:], sampling, [position + 1 + rows])
        return self._draft_draw(mixed, sampling, [position + 2 + r for r in range(rows)])

    def settle(self, cache: list[Any], keep: int, first: int, position: int, sampling: Any, count: int) -> list[int]:
        """After ``speculate``: forget the MTP entries of the rows past ``keep``, then the drafts for positions
        ``position``, ``position`` + 1, ...: the kept row's first draft ``first`` and ``count`` - 1 chained ones."""

        mtp_cache = cache[-1]
        out, rows = self._spec
        self._spec = None
        if rows > keep:
            mtp_cache.trim(rows - keep, self.args.indexer_compress_ratio)
        if count <= 0:
            return []
        streams = out[keep - 1:keep]
        if not self.queued_chains:
            drafts = [int(first)]
            for j in range(1, count):
                mixed, streams = self._mtp_step([drafts[-1]], streams, mtp_cache)
                mtp_cache.drafted += 1
                drafts.append(int(self._draft_draw(mixed, sampling, [position + j]).item()))
            return drafts
        if count == 1:
            return [int(first)]
        # each chained draft is drawn on the GPU and fed to the next step as an array: no read until the next
        # round's inputs are built
        chain = [mx.array([int(first)], dtype=mx.uint32)]
        for j in range(1, count):
            mixed, streams = self._mtp_step(chain[-1], streams, mtp_cache)
            mtp_cache.drafted += 1
            chain.append(self._draft_draw(mixed, sampling, [position + j]))
        drafts = mx.concatenate(chain)
        mx.async_eval(drafts)
        return drafts

    def unspeculate(self, cache: list[Any]) -> None:
        """Undo ``speculate`` entirely (the round's rows are absorbed another way)."""

        if getattr(self, "_spec", None) is not None:
            cache[-1].trim(self._spec[1], self.args.indexer_compress_ratio)
            self._spec = None

    # -- load-time check ------------------------------------------------------------------
    def check_windows(self, widest: int | None = None) -> tuple[int, dict[int, float]]:
        """The widest window (up to ``fused_rows``) whose every narrower window gives each row a one-row forward's
        logits bit for bit, from a 48-token prompt; and every exact width's forward time in ms (fastest of 3)."""

        import time

        from tensorfold.engine.lane_engine import LaneEngine

        copy = LaneEngine.copy_single_cache
        widest = int(widest or self.fused_rows)
        prompt = np.array([[(37 * i + 11) % 50_000 + 1000 for i in range(48)]], dtype=np.int64)
        window = [3001 + 17 * r for r in range(widest)]
        base = self.model.make_cache()
        mx.eval(self.model.hidden(prompt, base))
        one = copy(base)
        serial = []
        for token in window:
            logits = self.head(self.model.hidden(np.array([[token]], dtype=np.int64), one))
            mx.eval(logits)
            serial.append(logits[0, -1])
        exact = 1
        for width in range(2, widest + 1):
            logits = self.head(self.model.hidden(np.array([window[:width]], dtype=np.int64), copy(base)))
            mx.eval(logits)
            if not all(bool(mx.array_equal(logits[0, i], serial[i]).item()) for i in range(width)):
                break
            exact = width
        costs: dict[int, float] = {}
        for width in range(1, exact + 1):
            best = float("inf")
            for _ in range(3):
                cache = copy(base)
                started = time.perf_counter()
                mx.eval(self.head(self.model.hidden(np.array([window[:width]], dtype=np.int64), cache)))
                best = min(best, (time.perf_counter() - started) * 1e3)
            costs[width] = round(best, 3)
        return exact, costs


def load(model_dir: Path, *, drafts: int | None = None) -> tuple[FlashNext, Any]:
    """``drafts`` (default TF_FLASH_MTP, else 3; 0: none): the most drafts a round from the MTP head. The engine
    picks up to this many a round from the stream's recent acceptance: a chained draft step (~1.3 ms) and its
    verify row (~5 ms) pay only at high acceptance. On an M3 Ultra (2026-09-25) a 23k-token agent prompt with a
    512-token thinking budget, then a 2,550-token tool call, ran at 91.1 tok/s with a cap of 3 against 86.8 at 1."""

    import os

    from tensorfold.families.qwen4_exp import model as q4
    from tensorfold.families.qwen4_exp import mtp as mtp_module

    model, tokenizer = q4.load(Path(model_dir))
    drafts = int(os.environ.get("TF_FLASH_MTP", "3")) if drafts is None else int(drafts)
    head = mtp_module.load(Path(model_dir), model.args) if drafts > 0 and model.__dict__.get("fused") else None
    # TF_FLASH_MTP_GGUF=path: an MTP head from a llama.cpp GGUF (gguf_head.py) for a GGUF-trunk model (no fused path)
    gguf_head = os.environ.get("TF_FLASH_MTP_GGUF", "").strip()
    if head is None and drafts > 0 and gguf_head and not model.__dict__.get("fused"):
        from tensorfold.families.qwen4_exp.gguf_head import load_gguf_head

        head = load_gguf_head(gguf_head, model.args)
    return FlashNext(model, head, drafts=drafts), tokenizer
