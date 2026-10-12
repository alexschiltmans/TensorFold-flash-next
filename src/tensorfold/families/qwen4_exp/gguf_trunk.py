"""The trunk read in place from a llama.cpp GGUF file (mlx-kquant), instead of the checkpoint's MLX arrays.

A checkpoint whose config.json names a ``trunk_source`` takes every non-expert weight it can from the GGUF:

    "trunk_source": {"format": "gguf-kquant", "file": "/path/to/model-00001-of-00002.gguf",
                     "require_zero_copy": true, "fallback": []}

- K-quant and i-quant matrices (Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, IQ4_NL) become ``mlx_kquant.nn.KQuantLinear``
  (``token_embd``: ``KQuantEmbedding``) whose ``weight`` is the GGUF's own bytes, a no-copy view of the read-only
  mmap (clean file-backed pages, as the experts in gguf_experts.py).
- BF16 matrices (routers, hyper-connections, DeltaNet a/b, the indexer, PLE key/value) are plain ``nn.Linear``
  weights, also no-copy views.
- Small tensors are transcoded once (anonymous memory, about 4 MB at 48 layers): llama.cpp's converter stores the
  zero-centred norms as gamma = 1 + w (float32; here w = gamma - 1 again, float32), A_log as -exp(A_log) (here
  log(-a)), conv kernels as float32 or float16 (here cast to the checkpoint's bf16 when every value is a bf16
  value), and splits the indexer's fused q/k projection in two (here concatenated again).
- llama.cpp stores the DeltaNet's 48 value heads tiled over the 16 key heads (v head j reads k head j % 16), not
  grouped (j // 3) as the checkpoint: the rows of in_proj_qkv's V part, in_proj_z/a/b, A_log, dt_bias and the conv's
  V channels, and the columns of out_proj, are permuted. A column permutation of a K-quant matrix cannot be undone
  on its bytes, so the layer runs in the GGUF's order instead: ``GatedDeltaNet.v_tiled`` repeats q and k by tiling
  (the same math, an exact reindexing of heads). The recurrent state is then in tiled head order too.
- ``fallback``: regexes of TensorFold parameter paths to keep from the checkpoint (a DeltaNet layer is all or none).

Nothing here wires memory. The numpy-only parts (``trunk_plan``, the transforms) import no MLX, so a CPU check can
use them with gguf-py alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "gguf-kquant"
# ggml type id -> (name, kquant codec or None for a dense dtype)
GGML_TYPES = {
    0: ("F32", None), 1: ("F16", None), 30: ("BF16", None),
    8: ("Q8_0", "q8_0"), 12: ("Q4_K", "q4_k"), 13: ("Q5_K", "q5_k"), 14: ("Q6_K", "q6_k"),
    20: ("IQ4_NL", "iq4_nl"), 23: ("IQ4_XS", "iq4_xs"),
}


@dataclass(frozen=True)
class Entry:
    tf: str                 # TensorFold parameter path (after sanitize), e.g. model.layers.0.linear_attn.out_proj.weight
    gguf: tuple[str, ...]   # GGUF tensor name(s)
    transform: str          # none | norm_minus1 | a_log | conv | concat | row | cast
    vtiled: str = ""        # how llama.cpp permuted the DeltaNet's value heads: "" | rows_v | rows | cols | conv_v


def trunk_plan(num_layers: int, layer_types: list[str], ple_layers: list[int]) -> list[Entry]:
    """Every trunk parameter of an n-layer model and where it comes from in the GGUF.

    ``ple_layers``: 0-based indices of the layers with a PLE block (the config's ple_layer_ids are 1-based)."""

    plan = [
        Entry("model.embed_tokens.weight", ("token_embd.weight",), "none"),
        Entry("lm_head.weight", ("output.weight",), "none"),
        Entry("model.hyper_connection_mixer.hc_norm.weight", ("output_hc_norm.weight",), "norm_minus1"),
        Entry("model.hyper_connection_mixer.input_mix_weight_down.weight", ("output_hc_down.weight",), "none"),
        Entry("model.hyper_connection_mixer.input_mix_weight_up.weight", ("output_hc_up.weight",), "none"),
    ]
    for i in range(num_layers):
        p, b = f"model.layers.{i}", f"blk.{i}"
        for hc, g in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
            plan += [
                Entry(f"{p}.{hc}.hc_norm.weight", (f"{b}.{g}_norm.weight",), "norm_minus1"),
                Entry(f"{p}.{hc}.input_mix_weight_down.weight", (f"{b}.{g}_down.weight",), "none"),
                Entry(f"{p}.{hc}.input_mix_weight_up.weight", (f"{b}.{g}_up.weight",), "none"),
                Entry(f"{p}.{hc}.block_inject_weight.weight", (f"{b}.{g}_inject.weight",), "none"),
            ]
        plan += [
            Entry(f"{p}.mlp.gate.weight", (f"{b}.ffn_gate_inp.weight",), "none"),
            Entry(f"{p}.mlp.shared_expert_gate.weight", (f"{b}.ffn_gate_inp_shexp.weight",), "row"),
            Entry(f"{p}.mlp.shared_expert.gate_proj.weight", (f"{b}.ffn_gate_shexp.weight",), "none"),
            Entry(f"{p}.mlp.shared_expert.up_proj.weight", (f"{b}.ffn_up_shexp.weight",), "none"),
            Entry(f"{p}.mlp.shared_expert.down_proj.weight", (f"{b}.ffn_down_shexp.weight",), "none"),
        ]
        if layer_types[i] == "linear_attention":
            la = f"{p}.linear_attn"
            plan += [
                Entry(f"{la}.in_proj_qkv.weight", (f"{b}.attn_qkv.weight",), "none", "rows_v"),
                Entry(f"{la}.in_proj_z.weight", (f"{b}.attn_gate.weight",), "none", "rows"),
                Entry(f"{la}.in_proj_a.weight", (f"{b}.ssm_alpha.weight",), "none", "rows"),
                Entry(f"{la}.in_proj_b.weight", (f"{b}.ssm_beta.weight",), "none", "rows"),
                Entry(f"{la}.A_log", (f"{b}.ssm_a",), "a_log", "rows"),
                Entry(f"{la}.dt_bias", (f"{b}.ssm_dt.bias",), "cast", "rows"),
                Entry(f"{la}.conv1d.weight", (f"{b}.ssm_conv1d.weight",), "conv", "conv_v"),
                Entry(f"{la}.norm.weight", (f"{b}.ssm_norm.weight",), "cast"),
                Entry(f"{la}.out_proj.weight", (f"{b}.ssm_out.weight",), "none", "cols"),
            ]
        else:
            sa = f"{p}.self_attn"
            plan += [
                Entry(f"{sa}.q_proj.weight", (f"{b}.attn_q.weight",), "none"),
                Entry(f"{sa}.k_proj.weight", (f"{b}.attn_k.weight",), "none"),
                Entry(f"{sa}.v_proj.weight", (f"{b}.attn_v.weight",), "none"),
                Entry(f"{sa}.o_proj.weight", (f"{b}.attn_output.weight",), "none"),
                Entry(f"{sa}.q_norm.weight", (f"{b}.attn_q_norm.weight",), "norm_minus1"),
                Entry(f"{sa}.k_norm.weight", (f"{b}.attn_k_norm.weight",), "norm_minus1"),
                Entry(f"{sa}.indexer.index_qk_proj.weight", (f"{b}.indexer.q_proj.weight", f"{b}.indexer.k_proj.weight"),
                      "concat"),
                Entry(f"{sa}.indexer.q_layernorm.weight", (f"{b}.indexer.q_norm.weight",), "norm_minus1"),
                Entry(f"{sa}.indexer.k_layernorm.weight", (f"{b}.indexer.k_norm.weight",), "norm_minus1"),
            ]
        if i in ple_layers:
            pl = f"{p}.ple"
            plan += [
                Entry(f"{pl}.key_proj.weight", (f"{b}.ple_key.weight",), "none"),
                Entry(f"{pl}.value_proj.weight", (f"{b}.ple_value.weight",), "none"),
                Entry(f"{pl}.norm_key.weight", (f"{b}.ple_norm_key.weight",), "norm_minus1"),
                Entry(f"{pl}.norm_query.weight", (f"{b}.ple_norm_query.weight",), "norm_minus1"),
                Entry(f"{pl}.norm_conv.weight", (f"{b}.ple_norm_conv.weight",), "norm_minus1"),
                Entry(f"{pl}.conv1d.weight", (f"{b}.ple_conv1d.weight",), "conv"),
            ]
    return plan


def v_tiled_perm(n_rows: int, num_k_heads: int, num_v_heads: int, head_dim: int) -> np.ndarray:
    """Index p with gguf[r] = checkpoint[p[r]] for a V-head axis of n_rows = num_v_heads * head_dim entries
    (llama.cpp's _reorder_v_heads: grouped [k][r][d] -> tiled [r][k][d])."""

    per_k = num_v_heads // num_k_heads
    idx = np.arange(n_rows).reshape(num_k_heads, per_k, head_dim)
    return idx.transpose(1, 0, 2).reshape(-1)


def to_checkpoint_order(w: np.ndarray, vtiled: str, cfg: dict[str, int]) -> np.ndarray:
    """A dequantized GGUF tensor (numpy, checkpoint axis order) back in the checkpoint's V-head order: for checks
    only (the model runs in the GGUF's order)."""

    if not vtiled:
        return w
    nk, nv, dk, dv = cfg["nk"], cfg["nv"], cfg["dk"], cfg["dv"]
    if vtiled == "rows_v":
        qk = 2 * nk * dk
        perm = v_tiled_perm(nv * dv, nk, nv, dv)
        v = np.empty_like(w[qk:])
        v[perm] = w[qk:]
        return np.concatenate([w[:qk], v], axis=0)
    if vtiled == "conv_v":
        return to_checkpoint_order(w, "rows_v", cfg)
    if vtiled == "rows":
        head = w.shape[0] // nv
        perm = v_tiled_perm(w.shape[0], nk, nv, head)
        out = np.empty_like(w)
        out[perm] = w
        return out
    if vtiled == "cols":
        perm = v_tiled_perm(w.shape[1], nk, nv, dv)
        out = np.empty_like(w)
        out[:, perm] = w
        return out
    raise ValueError(vtiled)


def is_bf16_exact(a: np.ndarray) -> bool:
    """Whether every value of a float32/float16 array is a bf16 value (the cast to bf16 loses nothing)."""

    f = np.asarray(a, dtype=np.float32)
    return bool(np.all((f.view(np.uint32) & 0xFFFF) == 0))


class GGUFTrunk:
    """Plan, module swaps and arrays for one model's trunk from a GGUF file."""

    def __init__(self, model_dir: Path, source: dict[str, Any], cfg: Any) -> None:
        from tensorfold.families.qwen4_exp.gguf_experts import read_gguf_tensors

        if source.get("format") != FORMAT:
            raise ValueError(f"trunk_source format {source.get('format')!r}: only {FORMAT!r} is implemented")
        path = Path(str(source["file"])).expanduser()
        if not path.is_absolute():
            path = Path(model_dir) / path
        self.path = path.resolve()
        self.require_zero_copy = bool(source.get("require_zero_copy", True))
        fallback = [re.compile(p) for p in source.get("fallback", [])]
        self.tensors = read_gguf_tensors(self.path)
        ple = [i - 1 for i in cfg.ple_layer_ids]
        plan = trunk_plan(cfg.num_hidden_layers, list(cfg.layer_types), ple)
        self.fallback = [e.tf for e in plan if any(p.search(e.tf) for p in fallback)]
        self.plan = [e for e in plan if e.tf not in self.fallback]
        for e in self.plan:
            for name in e.gguf:
                t = self.tensors.get(name)
                if t is None:
                    raise ValueError(f"{self.path.name}: no {name} (for {e.tf})")
                if t.ggml_type not in GGML_TYPES:
                    raise ValueError(f"{name}: ggml type {t.ggml_type} is not handled here")
        self.cfg = cfg
        # DeltaNet layers in the GGUF's (tiled) value-head order: all of their tensors must come from the GGUF
        self.tiled_layers: list[int] = []
        for i, kind in enumerate(cfg.layer_types):
            if kind != "linear_attention":
                continue
            mine = [e for e in plan if e.tf.startswith(f"model.layers.{i}.linear_attn.") and e.vtiled]
            taken = [e for e in mine if e.tf not in self.fallback]
            if taken and len(taken) != len(mine):
                raise ValueError(f"layer {i}: a DeltaNet takes all of its head-ordered tensors from the GGUF or none "
                                 f"(fallback {sorted(set(e.tf for e in mine) - set(e.tf for e in taken))})")
            if taken:
                self.tiled_layers.append(i)

    # -- what is taken from where ----------------------------------------------------------------------------------
    def codec(self, entry: Entry) -> str | None:
        return GGML_TYPES[self.tensors[entry.gguf[0]].ggml_type][1]

    def replaces(self, key: str) -> bool:
        """A checkpoint key (after sanitize) this trunk replaces: the weight and its quantization arrays."""

        if "_keys" not in self.__dict__:
            self._keys = {e.tf for e in self.plan}
            self._modules = {e.tf[: -len(".weight")] for e in self.plan if e.tf.endswith(".weight")}
        return key in self._keys or key.rsplit(".", 1)[0] in self._modules

    def summary(self) -> dict[str, Any]:
        """GGUF tensor types taken, the fallback list, and (after ``arrays()``) the zero-copy/transcoded report."""

        kinds: dict[str, int] = {}
        for e in self.plan:
            for name in e.gguf:
                kind = GGML_TYPES[self.tensors[name].ggml_type][0]
                kinds[kind] = kinds.get(kind, 0) + 1
        report = self.__dict__.get("report") or {}
        return {"file": str(self.path), "entries": len(self.plan), "tensor_types": kinds, "fallback": self.fallback,
                "tiled_deltanet_layers": len(self.tiled_layers),
                "zero_copy_entries": len(report.get("zero_copy", [])), "zero_copy_bytes": report.get("zero_copy_bytes"),
                "transcoded_entries": len(report.get("transcoded", {})),
                "transcoded_bytes": report.get("transcoded_bytes")}

    # -- modules ---------------------------------------------------------------------------------------------------
    def install(self, model: Any) -> None:
        """Swap K-quant-backed Linear/Embedding modules in, before load_weights; mark the tiled DeltaNets."""

        from mlx_kquant.nn import KQuantEmbedding, KQuantLinear

        for e in self.plan:
            codec = self.codec(e)
            if codec is None or e.transform != "none":
                continue
            owner_path, _ = e.tf.rsplit(".", 1)
            parent_path, attr = owner_path.rsplit(".", 1) if "." in owner_path else ("", owner_path)
            parent = _resolve(model, parent_path)
            old = getattr(parent, attr)
            if e.gguf[0] == "token_embd.weight":
                rows, dims = old.weight.shape
                new = KQuantEmbedding(rows, dims, codec)
            else:
                out_dims, in_dims = old.weight.shape
                new = KQuantLinear(in_dims, out_dims, False, codec)
            setattr(parent, attr, new)
        for i in self.tiled_layers:
            model.layers[i].linear_attn.v_tiled = True

    # -- arrays ----------------------------------------------------------------------------------------------------
    def arrays(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """(parameters by TensorFold path, a report: which came zero-copy, which were transcoded and how)."""

        import mlx.core as mx
        import mlx_kquant as kq

        names = sorted({n for e in self.plan for n in e.gguf})
        wanted = set(names)
        before = kq.zero_copy_view_count()
        arrays, codecs, _meta, _shapes = kq.load_gguf(str(self.path), zero_copy=True,
                                                      skip=[n for n in self.tensors if n not in wanted])
        views = kq.zero_copy_view_count() - before
        codecs = dict(codecs)
        if self.require_zero_copy and views != len(names):
            raise RuntimeError(f"kq.load_gguf wrapped {views} of {len(names)} trunk tensors without a copy "
                               "(trunk_source.require_zero_copy)")
        out: dict[str, Any] = {}
        report: dict[str, Any] = {"zero_copy": [], "transcoded": {}, "views": views,
                                  "zero_copy_bytes": 0, "transcoded_bytes": 0}

        def note_zero_copy(e: Entry) -> None:
            report["zero_copy"].append(e.tf)
            report["zero_copy_bytes"] += sum(int(arrays[n].nbytes) for n in e.gguf)

        def note_transcoded(e: Entry, how: str, value: Any) -> None:
            report["transcoded"][e.tf] = how
            report["transcoded_bytes"] += int(value.nbytes)

        for e in self.plan:
            src = [arrays[n] for n in e.gguf]
            codec = self.codec(e)
            if codec is not None:
                if e.transform != "none":
                    raise ValueError(f"{e.tf}: transform {e.transform} on a {codec} tensor")
                if codecs.get(e.gguf[0]) != codec:
                    raise ValueError(f"{e.gguf[0]}: kquant reads codec {codecs.get(e.gguf[0])!r}, expected {codec!r}")
                prefix = e.tf[: -len(".weight")]
                out[e.tf] = src[0]
                out[f"{prefix}.scales"] = arrays[e.gguf[0][: -len(".weight")] + ".scales"]
                note_zero_copy(e)
                continue
            a = src[0]
            if e.transform == "none":
                out[e.tf] = a
                note_zero_copy(e)
            elif e.transform == "row":
                out[e.tf] = a.reshape(1, -1)                                   # a view of the same buffer
                note_zero_copy(e)
            elif e.transform == "norm_minus1":
                v = a.astype(mx.float32) - 1.0
                out[e.tf] = v
                note_transcoded(e, "gamma - 1 (float32)", v)
            elif e.transform == "a_log":
                v = mx.log(-a.astype(mx.float32))
                out[e.tf] = _to_bf16_if_exact(np, mx, v, e.tf, report, allow_round=True)
                note_transcoded(e, report["transcoded"].get(e.tf, "log(-a)"), out[e.tf])
            elif e.transform == "cast":
                out[e.tf] = _to_bf16_if_exact(np, mx, a, e.tf, report)
                note_transcoded(e, report["transcoded"].get(e.tf, "cast"), out[e.tf])
            elif e.transform == "conv":
                v = a.reshape(a.shape[0], a.shape[1], 1)                       # (C, K) -> (C, K, 1) as nn.Conv1d
                out[e.tf] = _to_bf16_if_exact(np, mx, v, e.tf, report)
                note_transcoded(e, report["transcoded"].get(e.tf, "conv"), out[e.tf])
            elif e.transform == "concat":
                v = mx.concatenate(src, axis=0)
                out[e.tf] = v
                note_transcoded(e, "concat " + " + ".join(e.gguf), v)
            else:
                raise ValueError(e.transform)
        mx.eval([v for k, v in out.items() if k in report["transcoded"]])
        return out, report


def _to_bf16_if_exact(np: Any, mx: Any, a: Any, name: str, report: dict[str, Any], allow_round: bool = False) -> Any:
    """The checkpoint keeps these in bf16: cast when that loses nothing (or, for A_log, round once; the check
    script compares the result with the original bf16 values)."""

    if a.dtype == mx.bfloat16:
        report["transcoded"][name] = "bf16 as stored"
        return a
    exact = is_bf16_exact(np.array(a.astype(mx.float32)))
    if exact or allow_round:
        report["transcoded"][name] = f"{a.dtype} -> bf16 ({'exact' if exact else 'rounded'})"
        return a.astype(mx.bfloat16)
    report["transcoded"][name] = f"kept {a.dtype} (not bf16-exact)"
    return a


def _resolve(model: Any, path: str) -> Any:
    obj = model
    for part in path.split(".") if path else []:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj
