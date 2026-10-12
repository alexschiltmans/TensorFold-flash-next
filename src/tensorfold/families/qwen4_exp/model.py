"""Qwen3.8 Flash Next (model_type ``qwen4_exp``): TensorFold's own forward pass.

Text only, written from the checkpoint layout:

- four residual streams joined by hyper-connections (each block reads a learned
  mix of the streams and writes back to each with its own gate);
- 36 Gated DeltaNet layers and 12 Qwen sparse-attention (QSA) layers, whose
  block indexer picks 512 blocks of 4 keys once the context passes 2,048 tokens;
- a 512-expert MoE (top 10) with a gated shared expert in every layer;
- a hashed 2- and 3-gram embedding (PLE, 128 row shards) added before layer 1.

Weights are the MLX 4-bit checkpoint as shipped (group size 32). The n-gram
hashes are computed on the host from token ids the caller already holds, so a
decode step never waits on the GPU to find its embedding rows.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.switch_layers import SwitchGLU

from tensorfold.families.qwen4_exp import sparse_prefill

MODEL_TYPE = "qwen4_exp"

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_PRIME = 10007

# head dims MLX 0.32 has fused prefill attention kernels for (steel_attention); any other head dim (Flash Next's
# 256) takes its unfused fallback, which materializes the [heads, rows, keys] score matrix in the input dtype
_FUSED_PREFILL_DIMS = (64, 80, 128)
# MLX 0.32 runs queries of at most this many rows through sdpa_vector (another kernel): row blocks stay wider
_VECTOR_ROWS = 8


def _env_int(name: str, default: int = 0) -> int:
    """An integer setting from the environment (unset or empty: ``default``)."""

    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _row_blocks(rows: int, block: int, offset: int = 0, align: int = 32) -> list[tuple[int, int]]:
    """[start, stop) row ranges of about ``block`` rows (rounded down to a multiple of ``align``, at least
    ``align``) for a call whose first row sits at position ``offset``. Every block but the last ends at a
    position that is a multiple of ``align``: a block's keys are cut at its end, and MLX's matmul sums a row
    exactly as the one call does only when that cut is aligned (blocks ending at 546 or 1,365 keys differed by up
    to 1e-4; multiples of 32 were bit-identical, MLX 0.32, M4 Pro). No block has ``_VECTOR_ROWS`` rows or fewer
    (a short first or last piece joins its neighbour), so every block takes the whole call's attention kernel."""

    block = max(align, block - block % align)
    cuts = list(range((offset // block + 1) * block - offset, rows, block))
    if cuts and cuts[0] <= _VECTOR_ROWS:
        cuts.pop(0)
    if cuts and rows - cuts[-1] <= _VECTOR_ROWS:
        cuts.pop()
    edges = [0, *cuts, rows]
    return list(zip(edges[:-1], edges[1:]))


@dataclass
class Config:
    hidden_size: int
    num_hidden_layers: int
    layer_types: list[str]
    vocab_size: int
    rms_norm_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rotary_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    output_gate_type: str
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    hc_count: int
    hc_lowrank: int
    indexer_n_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    ple_layer_ids: list[int]          # one-indexed decoder layers that get the n-gram embedding
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    ngram_vocab_divisor: int
    ngram_shards: int
    seed: int
    ple_eos: int
    group_size: int
    bits: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Config":
        t = dict(config.get("text_config") or config)
        rope = dict(t.get("rope_parameters") or {})
        head_dim = int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"])
        partial = float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
        eos = t.get("eos_token_id")
        quant = config.get("quantization") or config.get("quantization_config") or {}
        layer_types = [
            "linear_attention" if kind == "linear_attention" else "sparse_attention"
            for kind in t["layer_types"]
        ]
        return cls(
            hidden_size=int(t["hidden_size"]),
            num_hidden_layers=int(t["num_hidden_layers"]),
            layer_types=layer_types,
            vocab_size=int(t["vocab_size"]),
            rms_norm_eps=float(t["rms_norm_eps"]),
            num_attention_heads=int(t["num_attention_heads"]),
            num_key_value_heads=int(t["num_key_value_heads"]),
            head_dim=head_dim,
            rope_theta=float(rope.get("rope_theta", 10_000_000)),
            rotary_dim=int(head_dim * partial),
            linear_num_key_heads=int(t["linear_num_key_heads"]),
            linear_num_value_heads=int(t["linear_num_value_heads"]),
            linear_key_head_dim=int(t["linear_key_head_dim"]),
            linear_value_head_dim=int(t["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(t["linear_conv_kernel_dim"]),
            output_gate_type=str(t.get("output_gate_type") or "sigmoid"),
            num_experts=int(t["num_experts"]),
            num_experts_per_tok=int(t["num_experts_per_tok"]),
            moe_intermediate_size=int(t["moe_intermediate_size"]),
            shared_expert_intermediate_size=int(t["shared_expert_intermediate_size"]),
            hc_count=int(t.get("hc_count", 4)),
            hc_lowrank=int(t.get("hc_lowrank", 320)),
            indexer_n_heads=int(t.get("indexer_n_heads", 4)),
            indexer_head_dim=int(t.get("indexer_head_dim", 128)),
            indexer_budget=int(t.get("indexer_budget", 2048)),
            indexer_compress_ratio=int(t.get("indexer_compress_ratio", 4)),
            ple_layer_ids=sorted({int(i) for i in t.get("ple_layer_ids") or []}),
            ple_embed_dim=int(t.get("ple_embed_dim") or t["hidden_size"]),
            ple_conv_kernel_size=int(t.get("ple_conv_kernel_size", 4)),
            ngram_size=int(t.get("ngram_size", 3)),
            heads_per_ngram=int(t.get("heads_per_ngram", 8)),
            ngram_vocab_size_base=int(t.get("ngram_vocab_size_base", 20_000_000)),
            ngram_vocab_divisor=int(t.get("make_ngram_vocab_size_divisible_by", 128)),
            ngram_shards=int(t.get("split_ngram_parts", 128)),
            seed=int(t.get("seed", 1234)),
            ple_eos=int(eos[0] if isinstance(eos, list) else eos) if eos is not None else 0,
            group_size=int(quant.get("group_size", 32)),
            bits=int(quant.get("bits", 4)),
        )


# -- norms -----------------------------------------------------------------
def _derived(module: nn.Module, name: str, make: Any) -> mx.array:
    """A constant derived from a loaded weight, kept outside the parameter tree."""

    cache = module.__dict__.setdefault("_derived", {})
    value = cache.get(name)
    if value is None:
        value = make()
        mx.eval(value)
        cache[name] = value
    return value


class CenteredRMSNorm(nn.Module):
    """RMSNorm whose weight is stored centred on zero: scale by (1 + w), in float32.

    ``group`` normalizes each run of ``group`` features on its own (one per residual stream).
    """

    def __init__(self, dims: int, eps: float, group: int | None = None) -> None:
        super().__init__()
        self.eps = eps
        self.group = group
        self.weight = mx.zeros((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        scale = _derived(self, "scale", lambda: 1.0 + self.weight.astype(mx.float32))
        y = x.astype(mx.float32)
        if self.group is not None:
            y = y.reshape(*y.shape[:-1], -1, self.group)
            scale = scale.reshape(-1, self.group)
        y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + self.eps)
        return (y * scale).reshape(x.shape).astype(x.dtype)


class GatedRMSNorm(nn.Module):
    """RMSNorm of the recurrence output times a sigmoid (or SiLU) of the gate projection."""

    def __init__(self, dims: int, eps: float, activation: str) -> None:
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        y = mx.fast.rms_norm(x, self.weight, self.eps).astype(mx.float32)
        g = gate.astype(mx.float32)
        g = mx.sigmoid(g) if self.activation == "sigmoid" else nn.silu(g)
        return (y * g).astype(x.dtype)


# -- hyper-connections -------------------------------------------------------
class HyperConnection(nn.Module):
    """Read a block's input as a gated mix of the residual streams; say how much each stream takes back."""

    def __init__(self, cfg: Config, combine: bool = True) -> None:
        super().__init__()
        self.streams = cfg.hc_count
        self.dims = cfg.hidden_size
        wide = cfg.hc_count * cfg.hidden_size
        self.hc_norm = CenteredRMSNorm(wide, cfg.rms_norm_eps, group=cfg.hidden_size)
        self.input_mix_weight_down = nn.Linear(wide, cfg.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(cfg.hc_lowrank, wide, bias=False)
        if combine:
            self.block_inject_weight = nn.Linear(wide, cfg.hc_count, bias=False)

    def __call__(self, h: mx.array) -> Any:
        normed = self.hc_norm(h)
        mix = nn.silu(self.input_mix_weight_down(normed) / self.streams)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        shape = (*h.shape[:-1], self.streams, self.dims)
        mixed = mx.mean(mix.reshape(shape) * normed.reshape(shape), axis=-2)
        if "block_inject_weight" not in self:
            return mixed
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.streams)
        return mixed, inject


def _write_back(h: mx.array, branch: mx.array, inject: mx.array) -> mx.array:
    """Add the block's output to every residual stream, scaled by that stream's gate."""

    return h + (branch[..., None, :] * inject[..., None]).reshape(h.shape)


# -- caches ------------------------------------------------------------------
class LinearCache:
    """A DeltaNet layer: conv tail and recurrent state (plus the n-gram conv tail and token history on PLE layers)."""

    # rows of the last call on the PLE layer (history before it, its tokens, its conv input): not stored
    ple_rollback: Any = None
    transient = ("ple_rollback",)

    def __init__(self) -> None:
        self.conv: mx.array | None = None
        self.ssm: mx.array | None = None
        self.ple_conv: mx.array | None = None
        self.history: np.ndarray | None = None   # reassigned, never written in place (copies share it)
        self.offset = 0

    @property
    def state(self) -> list[mx.array]:
        return [a for a in (self.conv, self.ssm, self.ple_conv) if a is not None]


def cache_compact() -> bool:
    """TF_CACHE_COMPACT=1: each request reserves its room once (``AttentionCache.reserve``), growth past it is
    geometric, and stored copies hold exactly their positions (``AttentionCache.copy_compact``). MLX copies a
    buffer whole when an array another one still references is written: a resumed or snapshotted 32K cache did so
    at every request boundary and every 256 decode tokens, 2-4 GB wired spikes on the served model."""

    return os.environ.get("TF_CACHE_COMPACT", "").strip() not in ("", "0")


class AttentionCache:
    """A sparse-attention layer: keys, values, the indexer's raw keys, and its pooled block keys."""

    step = 256

    def __init__(self) -> None:
        self.keys: Any = None
        self.values: Any = None
        self.index_keys: Any = None
        self.pooled: Any = None     # normalized, rotated keys of blocks [0, pooled.shape[1])
        self.offset = 0
        self.reserved = 0           # positions the next allocation makes room for (TF_CACHE_COMPACT)

    def _room(self, end: int) -> int:
        """Capacity for ``end`` positions: the next multiple of ``step``, or with TF_CACHE_COMPACT at least the
        reservation and an eighth more than now (at least 4,096 positions), so a reply grows a buffer rarely."""

        want = end
        if cache_compact():
            want = max(end, self.reserved, self.offset + max(4096, self.offset // 8) if self.offset else end)
        return ((want + self.step - 1) // self.step) * self.step

    def _resize(self, cap: int) -> None:
        """Move positions [0, offset) into fresh buffers of ``cap`` positions (the arrays exist)."""

        n = self.offset

        def grow(old: mx.array, axis: int) -> mx.array:
            shape = list(old.shape)
            shape[axis] = cap
            if n == 0:
                return mx.zeros(tuple(shape), old.dtype)
            kept = old[:, :, :n] if axis == 2 else old[:, :n]
            shape[axis] = cap - n
            return mx.concatenate([kept, mx.zeros(tuple(shape), old.dtype)], axis=axis)

        self.keys = grow(self.keys, 2)
        self.values = grow(self.values, 2)
        self.index_keys = grow(self.index_keys, 1)

    def reserve(self, total: int) -> None:
        """Make room for ``total`` positions now or at the first write (TF_CACHE_COMPACT only)."""

        if not cache_compact():
            return
        self.reserved = max(self.reserved, int(total))
        if self.keys is not None and self.keys.shape[2] < total:
            self._resize(self._room(total))

    def copy_compact(self, reserve: int = 0) -> "AttentionCache":
        """A copy that shares no buffer with spare room: positions [0, offset) in buffers of exactly ``offset``
        positions (a stored entry), or of room for ``reserve`` (a request resumed from one). A full buffer is
        shared as it is: nothing writes past its end in place, growth allocates. Not evaluated here."""

        clone = type(self)()                      # a subclass (the MTP head's MTPCache) stays one
        if "drafted" in self.__dict__:
            clone.drafted = self.drafted
        clone.offset = n = self.offset
        clone.pooled = None if self.pooled is None else mx.array(self.pooled)
        if self.keys is None or n == 0:
            clone.reserved = max(self.reserved, int(reserve))
            return clone
        cap = ((max(n, int(reserve)) + self.step - 1) // self.step) * self.step if reserve > n else n

        def own(x: mx.array, axis: int) -> mx.array:
            if x.shape[axis] == n and cap == n:
                # full: its own handle on the same buffer (a rewrite after a rollback then copies, as MLX does
                # for any shared buffer; the same handle would change both caches)
                return mx.array(x)
            kept = x[:, :, :n] if axis == 2 else x[:, :n]
            if cap == n:
                return kept * mx.array(1, x.dtype)    # a new buffer: a view would keep the source's alive and shared
            pad = list(x.shape)
            pad[axis] = cap - n
            return mx.concatenate([kept, mx.zeros(tuple(pad), x.dtype)], axis=axis)

        clone.keys = own(self.keys, 2)
        clone.values = own(self.values, 2)
        clone.index_keys = own(self.index_keys, 1)
        return clone

    def trim(self, n: int, ratio: int = 4) -> int:
        """Forget the last ``n`` positions (and pooled blocks no longer complete)."""

        n = min(self.offset, n)
        self.offset -= n
        if self.pooled is not None and self.pooled.shape[1] > self.offset // ratio:
            blocks = self.offset // ratio
            self.pooled = self.pooled[:, :blocks] if blocks else None
        return n

    @property
    def state(self) -> list[mx.array]:
        if self.keys is None:
            return []
        n = self.offset
        out = [self.keys[:, :, :n], self.values[:, :, :n], self.index_keys[:, :n]]
        return out + ([self.pooled] if self.pooled is not None else [])

    def update(self, keys: mx.array, values: mx.array, index_keys: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        prev, length = self.offset, keys.shape[2]
        end = prev + length
        if self.keys is None:
            cap = self._room(end)
            batch, heads, _, dim = keys.shape
            self.keys = mx.zeros((batch, heads, cap, dim), keys.dtype)
            self.values = mx.zeros((batch, heads, cap, values.shape[3]), values.dtype)
            self.index_keys = mx.zeros((batch, cap, index_keys.shape[2]), index_keys.dtype)
        elif end > self.keys.shape[2]:
            self._resize(self._room(end))
        self.keys[:, :, prev:end] = keys
        self.values[:, :, prev:end] = values
        self.index_keys[:, prev:end] = index_keys
        self.offset = end
        return self.keys[:, :, :end], self.values[:, :, :end], self.index_keys[:, :end]


def _own_tail(x: mx.array, rows: int) -> mx.array:
    """The last ``rows`` rows of ``x`` [B, L, C] for a cache. A slice is a view that keeps all of ``x``'s buffer
    alive (a 2,048-row DeltaNet conv input is 42 MB, so 36 layers held 1.5 GB between prompt chunks); a call
    longer than a decode window gets a copy instead (same bits). TF_FLASH_TAIL_COPY=0: always the view."""

    tail = x[:, -rows:]
    if x.shape[1] > 2 * rows + 16 and os.environ.get("TF_FLASH_TAIL_COPY", "1") != "0":
        return mx.contiguous(tail)
    return tail


def _cache_state(cache: Any) -> list[mx.array]:
    """The arrays a layer's cache holds (whole buffers, no new slices)."""

    names = ("conv", "ssm", "ple_conv") if isinstance(cache, LinearCache) else ("keys", "values", "index_keys", "pooled")
    return [a for a in (getattr(cache, n, None) for n in names) if isinstance(a, mx.array)]


# -- Gated DeltaNet ------------------------------------------------------------
class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.nk, self.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = self.nk * self.dk, self.nv * self.dv
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kernel = cfg.linear_conv_kernel_dim
        d = cfg.hidden_size
        self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d, self.nv, bias=False)
        self.in_proj_a = nn.Linear(d, self.nv, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=self.kernel,
                                groups=self.conv_dim, bias=False)
        self.dt_bias = mx.ones((self.nv,))
        self.A_log = mx.zeros((self.nv,))
        self.norm = GatedRMSNorm(self.dv, cfg.rms_norm_eps, cfg.output_gate_type)
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)
        # True for weights in llama.cpp's value-head order (gguf_trunk.py): v head j reads k head j % nk, not j // r
        self.v_tiled = False

    def __call__(self, x: mx.array, cache: LinearCache) -> mx.array:
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
        # L2 normalization with the epsilon inside the sum (as the reference model), then 1/sqrt(d) on queries
        q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6) * (self.dk ** -0.5)
        k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
        if self.v_tiled and self.nv > self.nk:
            q = mx.tile(q, (1, 1, self.nv // self.nk, 1))
            k = mx.tile(k, (1, 1, self.nv // self.nk, 1))
        out, cache.ssm = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, cache.ssm)
        cache.offset += length
        out = self.norm(out, z)
        return self.out_proj(out.reshape(batch, length, -1))


# -- sparse attention ----------------------------------------------------------
class Indexer(nn.Module):
    """Scores 4-key blocks for each query; past 2,048 keys a query attends to its best 512 blocks and its tail."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.heads = cfg.indexer_n_heads
        self.dims = cfg.indexer_head_dim
        self.ratio = cfg.indexer_compress_ratio
        self.top_blocks = cfg.indexer_budget // cfg.indexer_compress_ratio
        self.rotary_dim = cfg.rotary_dim
        self.base = cfg.rope_theta
        self.index_qk_proj = nn.Linear(cfg.hidden_size, (self.heads + 1) * self.dims, bias=False)
        self.q_layernorm = CenteredRMSNorm(self.dims, cfg.rms_norm_eps)
        self.k_layernorm = CenteredRMSNorm(self.dims, cfg.rms_norm_eps)

    def project(self, x: mx.array) -> tuple[mx.array, mx.array]:
        batch, length, _ = x.shape
        qk = self.index_qk_proj(x).reshape(batch, length, self.heads + 1, self.dims)
        return qk[:, :, : self.heads], qk[:, :, self.heads]

    def _pool(self, raw: mx.array, start: int, stop: int) -> mx.array:
        """Mean of each block's raw keys, normalized, rotated to the block's first position."""

        batch = raw.shape[0]
        blocks = raw[:, start * self.ratio: stop * self.ratio].reshape(batch, stop - start, self.ratio, self.dims)
        pooled = mx.mean(blocks.astype(mx.float32), axis=-2).astype(raw.dtype)
        pooled = self.k_layernorm(pooled)
        return mx.fast.rope(pooled[:, None], self.rotary_dim, traditional=False, base=self.base,
                            scale=float(self.ratio), offset=start)[:, 0]

    def select(self, query: mx.array, raw: mx.array, cache: AttentionCache, past: int) -> mx.array | None:
        """Keys each query may read, [B, 1, L, keys] (bool), or None while the context is short (causal)."""

        batch, length = query.shape[0], query.shape[1]
        keys = past + length
        blocks = keys // self.ratio
        if blocks <= self.top_blocks:
            return None
        done = 0 if cache.pooled is None else cache.pooled.shape[1]
        if blocks > done:
            fresh = self._pool(raw, done, blocks)
            cache.pooled = fresh if cache.pooled is None else mx.concatenate([cache.pooled, fresh], axis=1)
        pooled = cache.pooled[:, :blocks]
        q = self.q_layernorm(query).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.rotary_dim, traditional=False, base=self.base, scale=1.0, offset=past)
        # float32 scores: which blocks win is a discrete choice and rounding flips the ones at the cut
        scores = q.astype(mx.float32) @ pooled.astype(mx.float32)[:, None].transpose(0, 1, 3, 2)
        scores = mx.sum(mx.maximum(scores, 0), axis=1) / math.sqrt(self.dims)          # [B, L, blocks]
        ends = past + mx.arange(length) + 1
        complete = ends // self.ratio
        valid = mx.arange(blocks)[None, None, :] < complete[None, :, None]
        scores = mx.where(valid, scores, -mx.inf)
        chosen = mx.argpartition(scores, kth=-self.top_blocks, axis=-1)[..., -self.top_blocks:]
        hits = mx.put_along_axis(mx.zeros((batch, length, blocks), dtype=mx.bool_), chosen,
                                 mx.array(True), axis=-1)
        picked = mx.repeat(hits, self.ratio, axis=-1)
        if blocks * self.ratio < keys:
            picked = mx.concatenate(
                [picked, mx.zeros((batch, length, keys - blocks * self.ratio), dtype=mx.bool_)], axis=-1)
        index = mx.arange(keys)
        tail = (index[None, None, :] >= (complete * self.ratio)[None, :, None]) & (index[None, None, :] < ends[None, :, None])
        causal = index[None, None, :] < ends[None, :, None]
        sparse = complete > self.top_blocks
        return mx.where(sparse[None, :, None], picked | tail, causal)[:, None]

    # -- the same selection a block of query rows at a time (TF_FLASH_ATTN_ROWS) ---------------------------------
    def prepare(self, query: mx.array, raw: mx.array, cache: AttentionCache, past: int) -> mx.array | None:
        """What ``select_rows`` needs for a call of ``query``'s rows: pools the blocks completed by this call (as
        ``select``) and returns the rotated float32 index queries [B, H, L, d], or None while the context is
        short (every row causal)."""

        length = query.shape[1]
        blocks = (past + length) // self.ratio
        if blocks <= self.top_blocks:
            return None
        done = 0 if cache.pooled is None else cache.pooled.shape[1]
        if blocks > done:
            fresh = self._pool(raw, done, blocks)
            cache.pooled = fresh if cache.pooled is None else mx.concatenate([cache.pooled, fresh], axis=1)
        q = self.q_layernorm(query).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.rotary_dim, traditional=False, base=self.base, scale=1.0, offset=past)
        return q.astype(mx.float32)

    def select_rows(self, q: mx.array, cache: AttentionCache, past: int, length: int, start: int,
                    stop: int) -> mx.array | None:
        """``select``'s mask for rows [start, stop) of a ``length``-row call after ``past`` keys, cut to the keys
        those rows can see ([B, 1, rows, past + stop], bool), or None when every one of them is causal.

        Row for row the same as ``select``: each row's block scores span all blocks of the call (so the top-k sees
        the same row), and the columns cut off lie past the row's causal end, where ``select`` is False too."""

        batch = q.shape[0]
        rows, end = stop - start, past + stop
        blocks = (past + length) // self.ratio
        ends = past + start + mx.arange(rows) + 1
        complete = ends // self.ratio
        if (past + stop) // self.ratio <= self.top_blocks:          # complete <= top_blocks on every row
            return None
        pooled = cache.pooled[:, :blocks]
        scores = q[:, :, start:stop] @ pooled.astype(mx.float32)[:, None].transpose(0, 1, 3, 2)
        scores = mx.sum(mx.maximum(scores, 0), axis=1) / math.sqrt(self.dims)          # [B, rows, blocks]
        valid = mx.arange(blocks)[None, None, :] < complete[None, :, None]
        scores = mx.where(valid, scores, -mx.inf)
        chosen = mx.argpartition(scores, kth=-self.top_blocks, axis=-1)[..., -self.top_blocks:]
        hits = mx.put_along_axis(mx.zeros((batch, rows, blocks), dtype=mx.bool_), chosen,
                                 mx.array(True), axis=-1)
        # a sparse row's chosen blocks end at or before its last complete block (< end // ratio)
        seen = end // self.ratio
        picked = mx.repeat(hits[..., :seen], self.ratio, axis=-1)
        if seen * self.ratio < end:
            picked = mx.concatenate([picked, mx.zeros((batch, rows, end - seen * self.ratio), dtype=mx.bool_)],
                                    axis=-1)
        index = mx.arange(end)
        tail = (index[None, None, :] >= (complete * self.ratio)[None, :, None]) & (index[None, None, :] < ends[None, :, None])
        causal = index[None, None, :] < ends[None, :, None]
        sparse = complete > self.top_blocks
        return mx.where(sparse[None, :, None], picked | tail, causal)[:, None]


class SparseAttention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.dims = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.scale = self.dims ** -0.5
        self.rotary_dim, self.base = cfg.rotary_dim, cfg.rope_theta
        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, self.heads * self.dims * 2, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads * self.dims, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads * self.dims, bias=False)
        self.o_proj = nn.Linear(self.heads * self.dims, d, bias=False)
        self.q_norm = CenteredRMSNorm(self.dims, cfg.rms_norm_eps)
        self.k_norm = CenteredRMSNorm(self.dims, cfg.rms_norm_eps)
        self.indexer = Indexer(cfg)

    def __call__(self, x: mx.array, cache: AttentionCache) -> mx.array:
        batch, length, _ = x.shape
        past = cache.offset
        q = self.q_proj(x).reshape(batch, length, self.heads, 2 * self.dims)
        queries, gate = q[..., : self.dims], q[..., self.dims:]
        gate = gate.reshape(batch, length, self.heads * self.dims)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(self.k_proj(x).reshape(batch, length, self.kv_heads, self.dims)).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(batch, length, self.kv_heads, self.dims).transpose(0, 2, 1, 3)
        queries = mx.fast.rope(queries, self.rotary_dim, traditional=False, base=self.base, scale=1.0, offset=past)
        keys = mx.fast.rope(keys, self.rotary_dim, traditional=False, base=self.base, scale=1.0, offset=past)
        index_query, index_key = self.indexer.project(x)
        keys, values, raw = cache.update(keys, values, index_key)
        out = self._attention(queries, keys, values, index_query, raw, cache, past)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out * mx.sigmoid(gate))

    def _attention(self, queries: mx.array, keys: mx.array, values: mx.array, index_query: mx.array, raw: mx.array,
                   cache: AttentionCache, past: int) -> mx.array:
        """Attention [B, H, L, D] of this call's rows (their keys already in ``cache``). TF_FLASH_SPARSE_PREFILL=1:
        prompt rows read only their indexer-selected keys (sparse_prefill.py), unless ``attend`` returns None
        (no row past the indexer's budget, below TF_FLASH_SPARSE_MIN_KEYS, batch > 1); decode steps and drafted
        windows (at most ``DECODE_ROWS`` rows) never reach it. Everything else takes the dense path."""

        if queries.shape[2] > sparse_prefill.DECODE_ROWS and sparse_prefill.enabled():
            out = sparse_prefill.attend(self, queries, index_query, raw, cache, past)
            if out is not None:
                return out
        return self._dense_attention(queries, keys, values, index_query, raw, cache, past)

    def _dense_attention(self, queries: mx.array, keys: mx.array, values: mx.array, index_query: mx.array,
                         raw: mx.array, cache: AttentionCache, past: int) -> mx.array:
        """MLX's attention under the indexer's mask over every key (in row blocks where TF_FLASH_ATTN_MB /
        TF_FLASH_ATTN_ROWS bound the score matrix)."""

        length = queries.shape[2]
        block = self._block_rows(past + length)
        if block > _VECTOR_ROWS and length > block and self.dims not in _FUSED_PREFILL_DIMS:
            return self._attend_in_blocks(queries, keys, values, index_query, raw, cache, past, block)
        mask = self.indexer.select(index_query, raw, cache, past)
        if mask is None and length > 1:
            mask = "causal"
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)

    def _block_rows(self, keys: int) -> int:
        """Query rows a prefill attention call takes at a time (0: all): TF_FLASH_ATTN_ROWS rows, and/or as many
        as keep one call's bf16 score matrix [heads, rows, keys] within TF_FLASH_ATTN_MB MiB (at least 16 rows);
        the smaller of the two when both are set."""

        rows = _env_int("TF_FLASH_ATTN_ROWS")
        budget = _env_int("TF_FLASH_ATTN_MB")
        if budget > 0:
            fit = max(2 * _VECTOR_ROWS, (budget << 20) // (self.heads * keys * 2))
            rows = min(rows, fit) if rows > 0 else fit
        return rows

    def _attend_in_blocks(self, queries: mx.array, keys: mx.array, values: mx.array, index_query: mx.array,
                          raw: mx.array, cache: AttentionCache, past: int, block: int) -> mx.array:
        """Prefill attention ``block`` query rows at a time (TF_FLASH_ATTN_ROWS), each block against the keys up to
        its own causal end, its indexer mask built for its rows only.

        MLX's attention fallback (head dim 256) holds a [heads, rows, keys] score matrix: 0.75 GiB for 2,048 rows
        at 8K keys, 12 GiB at 131K. Row blocks bound it to [heads, block, keys], and each block is evaluated
        before the next is queued, so command buffers never hold more than two blocks' scores. Bit-identical to
        the one call (rows are independent, every block takes the same kernel; checked on MLX 0.32, M4 Pro)."""

        length = queries.shape[2]
        q_index = self.indexer.prepare(index_query, raw, cache, past)
        outs: list[mx.array] = []
        for start, stop in _row_blocks(length, block, offset=past):
            end = past + stop
            mask = None if q_index is None else self.indexer.select_rows(q_index, cache, past, length, start, stop)
            out = mx.fast.scaled_dot_product_attention(
                queries[:, :, start:stop], keys[:, :, :end], values[:, :, :end], scale=self.scale,
                mask="causal" if mask is None else mask)
            mx.async_eval(out)
            if outs:
                mx.eval(outs[-1])
            outs.append(out)
        return mx.concatenate(outs, axis=2)


# -- MoE -----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dims: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden, bias=False)
        self.up_proj = nn.Linear(dims, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.hidden_size
        self.top_k = cfg.num_experts_per_tok
        self.gate = nn.Linear(d, cfg.num_experts, bias=False)          # stays bf16 in the checkpoint
        self.switch_mlp = SwitchGLU(d, cfg.moe_intermediate_size, cfg.num_experts)
        self.shared_expert = MLP(d, cfg.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(d, 1, bias=False)

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        keep = self.__dict__.get("expert_keep")
        if keep is None:
            probs = mx.softmax(self.gate(x), axis=-1, precise=True)
        else:
            # TF_EXPERT_KEEP / set_expert_keep(): dropped experts' logits are -inf before the softmax and the top-k,
            # so they get probability 0, are never chosen, and the kept experts' renormalized top-k weights equal a
            # router whose dropped rows were removed (the top-k renormalization cancels the softmax denominator)
            logits = self.gate(x)
            logits = mx.where(keep, logits, mx.array(-math.inf, dtype=logits.dtype))
            probs = mx.softmax(logits, axis=-1, precise=True)
        experts = mx.argpartition(probs, kth=-self.top_k, axis=-1)[..., -self.top_k:]
        weights = mx.take_along_axis(probs, experts, axis=-1)
        return experts, weights / weights.sum(axis=-1, keepdims=True)

    def __call__(self, x: mx.array) -> mx.array:
        experts, weights = self.route(x)
        inv = self.__dict__.get("expert_inv")
        if inv is not None:
            # experts_cut (set_expert_cut): the stacks hold only the kept experts, in ascending router-id order;
            # route() masks to them, and inv maps a router id to its storage index (monotone, so SwitchGLU's sorted
            # prefill path groups the same rows in the same order)
            if os.environ.get("TF_EXPERT_CUT_CHECK") == "1" and not bool(mx.all(mx.take(self.__dict__["expert_cut"],
                                                                                          experts)).item()):
                raise RuntimeError("experts_cut: a routed expert is outside the cut (TF_EXPERT_CUT_CHECK)")
            experts = mx.take(inv, experts)
        routed = (self.switch_mlp(x, experts) * weights[..., None]).sum(axis=-2)
        shared = self.shared_expert(x) * mx.sigmoid(self.shared_expert_gate(x))
        return routed + shared


def set_expert_keep(model: Any, keep: dict[str, Any] | None) -> int:
    """Mask routed experts out of the routers (an experiment switch, default off): ``keep`` maps a layer index
    (str or int) to the router-local expert indices that stay routable, or carries such a map under
    ``"keep_local"``; layers not in it, or ``keep=None``, route over all experts again (the original code path,
    bit-identical). The mask acts on the router logits before the softmax and the top-k, so routing equals a router
    with the dropped rows removed; the shared expert and its gate are untouched. Returns the masked layer count."""

    if model.__dict__.get("fused") is not None:
        raise ValueError("expert masks need the reference forward (TF_FLASH_FUSED=0): the fused decode routes itself")
    if keep is not None and "keep_local" in keep:
        keep = keep["keep_local"]
    keep = {int(k): v for k, v in (keep or {}).items()}
    n = 0
    for i, layer in enumerate(model.layers):
        moe = layer.mlp
        cut = moe.__dict__.get("expert_cut_np")      # experts_cut: the mask never goes below the cut
        if i not in keep:
            if cut is None:
                moe.__dict__.pop("expert_keep", None)
            else:
                moe.__dict__["expert_keep"] = moe.__dict__["expert_cut"]
            continue
        experts = int(moe.gate.weight.shape[0]) if hasattr(moe.gate, "weight") else None
        ids = np.asarray(keep[i], dtype=np.int64)
        size = experts or int(ids.max()) + 1
        if ids.size < moe.top_k or ids.min() < 0 or ids.max() >= size:
            raise ValueError(f"layer {i}: {ids.size} kept experts, ids outside 0..{size - 1} or fewer than top-k")
        mask = np.zeros(size, dtype=bool)
        mask[ids] = True
        if cut is not None and bool((mask & ~cut).any()):
            raise ValueError(f"layer {i}: the mask keeps {int((mask & ~cut).sum())} experts that experts_cut dropped "
                             "(their weights are not loaded)")
        moe.__dict__["expert_keep"] = mx.array(mask)
        n += 1
    return n


def set_expert_cut(model: Any, keep_local: dict[str, Any]) -> list[int]:
    """experts_cut (config.json, default off): layer L's expert stacks hold only router ids ``keep_local[L]``, in
    ascending order. Stores per layer (out of the module tree, as ``expert_keep``): the cut as a router mask
    (``expert_cut``, also the default ``expert_keep``, so route() is the TF_EXPERT_KEEP code path with that mask) and
    ``expert_inv``: router id -> storage index, dropped ids -> 0 (a valid index: if routing ever escaped the mask, a
    gather reads a real expert, never out of bounds; TF_EXPERT_CUT_CHECK=1 asserts it never does). Every layer must
    be listed. set_expert_keep() then masks within the cut only. Returns the per-layer counts."""

    if model.__dict__.get("fused") is not None:
        raise ValueError("experts_cut needs the reference forward (TF_FLASH_FUSED=0): the fused decode routes itself")
    keep_local = {int(k): v for k, v in keep_local.items()}
    counts = []
    for i, layer in enumerate(model.layers):
        moe = layer.mlp
        if i not in keep_local:
            raise ValueError(f"experts_cut: no keep list for layer {i}")
        size = int(moe.gate.weight.shape[0])
        ids = np.asarray(keep_local[i], dtype=np.int64)
        if (ids.size < moe.top_k or ids.min() < 0 or ids.max() >= size or not bool(np.all(np.diff(ids) > 0))):
            raise ValueError(f"experts_cut layer {i}: ids must be ascending, unique, within 0..{size - 1}, at least "
                             "top-k of them")
        stored = int(moe.switch_mlp.gate_proj["weight"].shape[0])
        if stored != ids.size:
            raise ValueError(f"experts_cut layer {i}: {ids.size} kept ids, but the expert stacks hold {stored}")
        cut = np.zeros(size, dtype=bool)
        cut[ids] = True
        inv = np.zeros(size, dtype=np.uint32)
        inv[ids] = np.arange(ids.size, dtype=np.uint32)
        moe.__dict__["expert_cut_np"] = cut
        moe.__dict__["expert_cut"] = mx.array(cut)
        moe.__dict__["expert_inv"] = mx.array(inv)
        moe.__dict__["expert_keep"] = moe.__dict__["expert_cut"]
        counts.append(int(ids.size))
    return counts


def read_expert_cut(model_dir: Path, spec: dict[str, Any], num_layers: int) -> dict[int, list[int]]:
    """experts_cut's keep lists (``{"file": "cut_experts.json"}``, relative to the checkpoint dir or absolute):
    ``keep_local`` maps every layer to its kept router ids. A missing or short file fails the load."""

    path = Path(str(spec["file"])).expanduser()
    if not path.is_absolute():
        path = Path(model_dir) / path
    data = json.loads(path.read_text())
    keep = {int(k): sorted(int(e) for e in v) for k, v in data["keep_local"].items()}
    missing = [i for i in range(num_layers) if i not in keep]
    if missing:
        raise ValueError(f"experts_cut {path}: no keep list for layers {missing[:4]}")
    return keep


# -- n-gram embedding (PLE) ------------------------------------------------------
def _splitmix64(value: int) -> int:
    value = (value + _GOLDEN) & _MASK64
    value = ((value ^ (value >> 30)) * _MIX1) & _MASK64
    value = ((value ^ (value >> 27)) * _MIX2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def layer_multipliers(vocab: int, ngram: int, ple_index: int, seed: int) -> np.ndarray:
    half = max(1, (((1 << 63) - 1) // max(vocab, 1)) // 2)
    base = seed + _PRIME * ple_index
    return np.array([2 * (_splitmix64((base + _GOLDEN * (i + 1)) & _MASK64) % half) + 1 for i in range(ngram)],
                    dtype=np.int64)


class NGramEmbedding(nn.Module):
    """2- and 3-gram ids hashed into 16 heads of prime-sized tables, looked up in 128 row shards."""

    def __init__(self, cfg: Config, ple_index: int) -> None:
        super().__init__()
        self.n = cfg.ngram_size
        self.context = cfg.ngram_size - 1
        self.per_ngram = cfg.heads_per_ngram
        self.heads = self.context * self.per_ngram
        self.eos = cfg.ple_eos
        sizes, offsets, total = [], [], 0
        for head in range(self.heads):
            size = _nth_prime_after(cfg.ngram_vocab_size_base - 1, ple_index * self.heads + head + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        self.head_sizes = np.array(sizes, dtype=np.int64)
        self.head_offsets = np.array(offsets, dtype=np.int64)
        self.multipliers = layer_multipliers(cfg.vocab_size, cfg.ngram_size, ple_index, cfg.seed)
        rows = math.ceil(total / cfg.ngram_vocab_divisor) * cfg.ngram_vocab_divisor
        base, extra = divmod(rows, cfg.ngram_shards)
        shard_rows = [base + (1 if i < extra else 0) for i in range(cfg.ngram_shards)]
        self.shard_starts = [0]
        for count in shard_rows:
            self.shard_starts.append(self.shard_starts[-1] + count)
        self.dims = cfg.ple_embed_dim // self.heads
        self.shards = [nn.Embedding(count, self.dims) for count in shard_rows]
        # set by load() for a checkpoint whose config names an ``ngram_source``: the table is read from that file
        # (gguf_rows.GGUFRows) and ``shards`` is emptied, so the 51B-parameter table never enters the GPU's memory
        self.table = None

    def ids(self, history: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """Row ids [B, L, heads] for ``tokens`` [B, L] after ``history`` [B, n-1] (EOS resets the n-grams)."""

        seq = np.concatenate([history, tokens], axis=1).astype(np.int64)
        batch, width = seq.shape
        pos = np.arange(width)
        eos_at = np.where(seq == self.eos, pos[None], -1)
        before = np.concatenate([np.full((batch, 1), -1), np.maximum.accumulate(eos_at, axis=1)[:, :-1]], axis=1)
        in_segment = pos[None] - (before + 1)
        shifted = []
        for shift in range(self.n):
            source = pos - shift
            taken = np.take_along_axis(seq, np.broadcast_to(np.maximum(source, 0)[None], seq.shape), axis=1)
            shifted.append(np.where((in_segment >= shift) & (source[None] >= 0), taken, self.eos))
        blocks = []
        for ngram in range(2, self.n + 1):
            first = (ngram - 2) * self.per_ngram
            mixed = shifted[0] * self.multipliers[0]
            for p in range(1, ngram):
                mixed = np.bitwise_xor(mixed, shifted[p] * self.multipliers[p])
            sizes = self.head_sizes[first:first + self.per_ngram]
            blocks.append(mixed[..., None] % sizes + self.head_offsets[first:first + self.per_ngram])
        return np.concatenate(blocks, axis=-1)[:, -tokens.shape[1]:]

    def __call__(self, ids: np.ndarray) -> mx.array:
        if self.table is not None:
            return self.table(ids).reshape(*ids.shape[:-1], self.heads * self.dims)
        flat = ids.reshape(-1)
        shard = np.searchsorted(np.asarray(self.shard_starts), flat, side="right") - 1
        parts, order = [], []
        for s in np.unique(shard):
            where = np.nonzero(shard == s)[0]
            local = mx.array((flat[where] - self.shard_starts[int(s)]).astype(np.int32))
            parts.append(self.shards[int(s)](local))
            order.append(where)
        rows = mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        inverse = np.empty(len(flat), dtype=np.int32)
        inverse[np.concatenate(order)] = np.arange(len(flat), dtype=np.int32)
        rows = rows[mx.array(inverse)]
        return rows.reshape(*ids.shape[:-1], self.heads * self.dims)


class PLELayer(nn.Module):
    """Adds a gated n-gram embedding to every residual stream, then a dilated short conv over it."""

    def __init__(self, cfg: Config, ple_index: int) -> None:
        super().__init__()
        self.streams, self.dims = cfg.hc_count, cfg.hidden_size
        wide = cfg.hc_count * cfg.hidden_size
        self.ple_embedding = NGramEmbedding(cfg, ple_index)
        self.key_proj = nn.Linear(cfg.ple_embed_dim, wide, bias=False)
        self.value_proj = nn.Linear(cfg.ple_embed_dim, cfg.hidden_size, bias=False)
        self.norm_key = CenteredRMSNorm(wide, cfg.rms_norm_eps, group=cfg.hidden_size)
        self.norm_query = CenteredRMSNorm(wide, cfg.rms_norm_eps, group=cfg.hidden_size)
        self.norm_conv = CenteredRMSNorm(wide, cfg.rms_norm_eps, group=cfg.hidden_size)
        self.dilation = cfg.ngram_size
        self.tail = (cfg.ple_conv_kernel_size - 1) * self.dilation
        self.conv1d = nn.Conv1d(wide, wide, kernel_size=cfg.ple_conv_kernel_size, dilation=self.dilation,
                                groups=wide, bias=False)

    def _gated(self, h: mx.array, emb: mx.array) -> tuple[mx.array, mx.array]:
        """Rows of the gated n-gram embedding and their conv-norm: (gated, normed), each shaped as ``h``."""

        batch, length, _ = h.shape
        shape = (batch, length, self.streams, self.dims)
        keys = self.norm_key(self.key_proj(emb)).reshape(shape)
        values = self.value_proj(emb)
        queries = self.norm_query(h).reshape(shape)
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(self.dims)
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = (mx.sigmoid(gate) * values[..., None, :]).reshape(h.shape)
        return gated, self.norm_conv(gated)

    def __call__(self, h: mx.array, tokens: np.ndarray, cache: LinearCache) -> mx.array:
        batch, length, _ = h.shape
        history = cache.history
        if history is None:
            history = np.full((batch, self.ple_embedding.context), self.ple_embedding.eos, dtype=np.int64)
        ids = self.ple_embedding.ids(history, tokens)
        cache.history = np.concatenate([history, tokens.astype(np.int64)], axis=1)[:, -self.ple_embedding.context:]
        emb = self.ple_embedding(ids)
        block = _env_int("TF_FLASH_PLE_ROWS")
        if block > 0 and length > block:
            # TF_FLASH_PLE_ROWS=n: the per-row part (projections, three float32 norms, the gate) n rows at a time,
            # each block evaluated before the next is queued: at 2,048 rows it held ~1 GiB at once
            parts: list[tuple[mx.array, mx.array]] = []
            for start in range(0, length, block):
                pair = self._gated(h[:, start:start + block], emb[:, start:start + block])
                mx.async_eval(*pair)
                if parts:
                    mx.eval(*parts[-1])
                parts.append(pair)
            gated = mx.concatenate([g for g, _ in parts], axis=1)
            normed = mx.concatenate([n for _, n in parts], axis=1)
        else:
            gated, normed = self._gated(h, emb)
        tail = cache.ple_conv if cache.ple_conv is not None else mx.zeros((batch, self.tail, h.shape[-1]), h.dtype)
        conv_in = mx.concatenate([tail, normed], axis=1)
        cache.ple_conv = _own_tail(conv_in, self.tail)
        # what keeping only the first rows of this call needs: the history before it, its tokens, the conv window
        cache.ple_rollback = (history, tokens.astype(np.int64), conv_in)
        return gated + nn.silu(self.conv1d(conv_in))


# -- model -----------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config, index: int) -> None:
        super().__init__()
        self.is_linear = cfg.layer_types[index] == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            self.self_attn = SparseAttention(cfg)
        self.mlp = SparseMoE(cfg)
        if index + 1 in cfg.ple_layer_ids:
            self.ple = PLELayer(cfg, cfg.ple_layer_ids.index(index + 1))
        self.attn_hyper_connection = HyperConnection(cfg)
        self.mlp_hyper_connection = HyperConnection(cfg)

    def __call__(self, h: mx.array, tokens: np.ndarray, cache: Any) -> mx.array:
        for part in self.parts(tokens, cache):
            h = part(h)
        return h

    def parts(self, tokens: np.ndarray, cache: Any) -> list[Any]:
        """The layer as consecutive functions of the residual streams (the n-gram embedding on its layer, the
        mixing branch, the MoE), so a caller can hand them to the GPU one at a time (TF_FLASH_INFLIGHT_PARTS)."""

        def ple(h: mx.array) -> mx.array:
            return h + self.ple(h, tokens, cache)

        def mixing(h: mx.array) -> mx.array:
            mixed, inject = self.attn_hyper_connection(h)
            branch = self.linear_attn(mixed, cache) if self.is_linear else self.self_attn(mixed, cache)
            return _write_back(h, branch, inject)

        def moe(h: mx.array) -> mx.array:
            mixed, inject = self.mlp_hyper_connection(h)
            return _write_back(h, self.mlp(mixed), inject)

        return ([ple] if "ple" in self else []) + [mixing, moe]


class Body(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.hyper_connection_mixer = HyperConnection(cfg, combine=False)


class Qwen4Exp(nn.Module):
    # layers per slice handed to the GPU while Python builds the next slice (same graph, same bits)
    pipeline_layers = 4
    # decode steps of up to this many rows go through ``decode.FusedDecode`` when it is attached
    fused_rows = 16

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.args = cfg
        self.model = Body(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        return [LinearCache() if layer.is_linear else AttentionCache() for layer in self.layers]

    def hidden(self, inputs: Any, cache: list[Any]) -> mx.array:
        """The mixed hidden state [B, L, D] after the last layer."""

        tokens = np.asarray(inputs, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        fused = self.__dict__.get("fused")
        if fused is not None and tokens.shape[0] == 1 and tokens.shape[1] <= self.fused_rows:
            return fused(tokens, cache)
        h = self.model.embed_tokens(mx.array(tokens.astype(np.int32)))
        h = mx.tile(h, (1, 1, self.args.hc_count))
        # TF_FLASH_INFLIGHT=n (prompt chunks only): hand every layer to the GPU as it is built and wait for the
        # layer n back, so at most n + 1 layers' buffers are alive at once. Without it the whole chunk's graph is
        # queued ahead of the GPU and the command buffers keep every op's inputs until they complete (most
        # of the reference forward's transient memory). The layer's cache arrays go with it: their lazy slices
        # would otherwise keep each layer's whole conv input alive until the next call.
        # TF_FLASH_INFLIGHT_PARTS=1: the unit is a part of a layer (n-gram embedding, mixing branch, MoE), not a
        # whole layer.
        inflight = _env_int("TF_FLASH_INFLIGHT") if tokens.shape[1] > self.fused_rows else 0
        by_parts = inflight > 0 and _env_int("TF_FLASH_INFLIGHT_PARTS") > 0
        queued: list[mx.array] = []

        def queue(h: mx.array, layer_cache: Any) -> None:
            mx.async_eval(h, *_cache_state(layer_cache))
            queued.append(h)
            if len(queued) > inflight:
                mx.eval(queued.pop(0))

        for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            if by_parts:
                for part in layer.parts(tokens, layer_cache):
                    h = part(h)
                    queue(h, layer_cache)
                continue
            h = layer(h, tokens, layer_cache)
            if inflight > 0:
                queue(h, layer_cache)
            elif self.pipeline_layers and (i + 1) % self.pipeline_layers == 0:
                mx.async_eval(h)
        self.__dict__["last_streams"] = h[0]          # [L, S*D]: the residual streams before the final mixer
        return self.model.hyper_connection_mixer(h)

    def head(self, hidden: mx.array) -> mx.array:
        return self.lm_head(hidden)

    def __call__(self, inputs: Any, cache: list[Any]) -> mx.array:
        return self.head(self.hidden(inputs, cache))


# -- loading ---------------------------------------------------------------------
# hashing constants the checkpoint ships; each must equal what NGramEmbedding derives
_PLE_CONSTANTS = {
    "layer_multipliers": "multipliers",
    "ngram_heads_vocab_sizes": "head_sizes",
    "ngram_heads_offsets": "head_offsets",
}


def sanitize(weights: dict[str, mx.array]) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
    """Checkpoint names -> this module's; the n-gram hashing constants come back separately (not weights)."""

    out: dict[str, mx.array] = {}
    extras: dict[str, mx.array] = {}
    for name, value in weights.items():
        if not name.startswith("language_model.") or ".mtp." in name or name.startswith("language_model.mtp"):
            continue
        key = name[len("language_model."):]
        if key.rsplit(".", 1)[-1] in _PLE_CONSTANTS:
            extras[key] = value
            continue
        key = key.replace("ngram_embedding.shard_", "shards.")
        out[key] = value
    return out, extras


def norms_stored_around_one(weights: dict[str, mx.array]) -> bool:
    """Whether the checkpoint's centred norms hold gamma (around 1) instead of gamma - 1 (around 0).

    Decided from the attention hyper-connection norms, which sit near their centre in every layer
    (the reference implementation's rule: 90% of their means above 0.5 and a median in [0.75, 1.5]).
    """

    anchors = [value for key, value in weights.items()
               if key.startswith("model.layers.") and key.endswith(".attn_hyper_connection.hc_norm.weight")]
    if len(anchors) < 8:
        return False
    means = np.array([float(mx.mean(a.astype(mx.float32)).item()) for a in anchors])
    around_one = (means > 0.5).mean() >= 0.9 and 0.75 <= float(np.median(means)) <= 1.5
    around_zero = (means > 0.5).mean() <= 0.1 and -0.5 <= float(np.median(means)) <= 0.25
    if not (around_one or around_zero):
        raise ValueError(f"cannot tell how the norm weights are stored (median mean {np.median(means):.3f})")
    return around_one


def load(model_dir: Path, *, lazy: bool = False) -> tuple[Qwen4Exp, Any]:
    from mlx_lm.utils import load_tokenizer

    config = json.loads((Path(model_dir) / "config.json").read_text())
    cfg = Config.from_dict(config)
    model = Qwen4Exp(cfg)
    weights: dict[str, mx.array] = {}
    for path in sorted(Path(model_dir).glob("model*.safetensors")):
        weights.update(mx.load(str(path)))
    weights, extras = sanitize(weights)
    quant = config.get("quantization") or config.get("quantization_config") or {}
    source = config.get("ngram_source")
    if source:
        if source.get("format") != "gguf":
            raise ValueError(f"ngram_source format {source.get('format')!r}: only 'gguf' is implemented")
        from tensorfold.families.qwen4_exp.gguf_rows import GGUFRows

        for layer in model.layers:
            if "ple" in layer:
                embedding = layer.ple.ple_embedding
                embedding.shards = []
                embedding.table = GGUFRows.from_manifest(Path(model_dir), source["tables"], embedding.dims)

    experts = config.get("experts_source")
    cut_spec = config.get("experts_cut")
    cut_keep = None
    if cut_spec:
        # a cut checkpoint (set_expert_cut): its experts_source holds only the kept experts per layer
        if not experts:
            raise ValueError("experts_cut needs experts_source (the cut's expert file)")
        cut_keep = read_expert_cut(Path(model_dir), cut_spec, cfg.num_hidden_layers)
    gguf_experts = None
    if experts:
        # routed experts read in place from a GGUF (gguf_experts.py): the checkpoint's own expert arrays leave
        # mx.load's lazy dict here, before anything evaluates them, so they are never read from disk
        from tensorfold.families.qwen4_exp.gguf_experts import GGUFExperts, is_expert_key

        gguf_experts = GGUFExperts(Path(model_dir), experts, cfg.num_hidden_layers, cfg.num_experts,
                                   cfg.hidden_size, cfg.moe_intermediate_size,
                                   counts=None if cut_keep is None else
                                   [len(cut_keep[i]) for i in range(cfg.num_hidden_layers)])
        weights = {k: v for k, v in weights.items() if not is_expert_key(k)}
        gguf_experts.install(model)

    def quantized(path: str, module: nn.Module) -> bool | dict[str, Any]:
        if not (hasattr(module, "to_quantized") and f"{path}.scales" in weights):
            return False
        # per-module settings (mlx-lm's layout: the module's checkpoint path as a key) win over the global ones
        spec = quant.get(f"language_model.{path}")
        if isinstance(spec, dict):
            return {"group_size": int(spec["group_size"]), "bits": int(spec["bits"]),
                    "mode": str(spec.get("mode", "affine"))}
        return True

    trunk = config.get("trunk_source")
    gguf_trunk = None
    if trunk:
        # the trunk read in place from a GGUF too (gguf_trunk.py): K-quant matrices as kquant modules over the
        # file's bytes, bf16 matrices as no-copy views, small tensors transcoded; the checkpoint's arrays for them
        # leave the lazy dict unread, as the experts' do
        from tensorfold.families.qwen4_exp.gguf_trunk import GGUFTrunk

        gguf_trunk = GGUFTrunk(Path(model_dir), trunk, cfg)
        weights = {k: v for k, v in weights.items() if not gguf_trunk.replaces(k)}
        gguf_trunk.install(model)

    nn.quantize(model, group_size=cfg.group_size, bits=cfg.bits, class_predicate=quantized)
    if gguf_experts is not None:
        weights.update(gguf_experts.arrays())
        model.__dict__["gguf_experts"] = gguf_experts          # kept out of the module tree, as ``fused``
    if gguf_trunk is not None:
        trunk_arrays, gguf_trunk.report = gguf_trunk.arrays()
        weights.update(trunk_arrays)
        model.__dict__["gguf_trunk"] = gguf_trunk
    if norms_stored_around_one(weights):
        # early MLX conversions store these norms' gamma itself (around 1), not gamma - 1
        for path, module in model.named_modules():
            if isinstance(module, CenteredRMSNorm) and f"{path}.weight" in weights:
                weights[f"{path}.weight"] = weights[f"{path}.weight"].astype(mx.float32) - 1.0
    model.load_weights(list(weights.items()), strict=True)
    for key, value in extras.items():
        embedding = model.layers[int(key.split(".")[2])].ple.ple_embedding
        derived = getattr(embedding, _PLE_CONSTANTS[key.rsplit(".", 1)[-1]])
        shipped = np.array(value).astype(np.int64)
        if not np.array_equal(shipped, derived):
            raise ValueError(f"{key}: checkpoint {shipped} != derived {derived}")
    if not lazy:
        mx.eval(model.parameters())
        import os

        wired_gib = float(os.environ.get("TF_WIRED_GIB", "0"))
        if wired_gib > 0:
            # opt-in only. Wired memory cannot be paged out: once, a limit at the full recommended working
            # set (37.4 GiB) with a 34 GB build, plus the reference forward's large transient buffers at 8-32K
            # depth (MLX peak 46 GiB at 8K), left the 48 GB Mac without memory for the kernel and crashed it.
            # Without a limit macOS compresses MLX's buffers under pressure instead (slow, not fatal).
            mx.set_wired_limit(int(wired_gib * 2**30))

        # the fused kernels read 4-bit group-32 weights everywhere and the n-gram shards in memory
        fusable = (not source and not experts and not trunk and cfg.bits == 4 and cfg.group_size == 32
                   and not any(isinstance(v, dict) for v in quant.values()))
        if os.environ.get("TF_FLASH_FUSED", "1") != "0" and fusable:
            from tensorfold.families.qwen4_exp.decode import FusedDecode

            # kept out of the module tree (a plain attribute), so parameters() stays the checkpoint's
            try:
                model.__dict__["fused"] = FusedDecode(model)
            except ValueError as err:
                import sys

                print(f"tensorfold: fused decode off ({err}); using the reference forward", file=sys.stderr)
    if cut_keep is not None:
        counts = set_expert_cut(model, cut_keep)
        print(f"tensorfold: experts_cut: {sum(counts)} experts in {len(counts)} layers ({min(counts)}-{max(counts)} "
              "a layer)", flush=True)
    keep_path = os.environ.get("TF_EXPERT_KEEP", "").strip()
    if keep_path:
        n = set_expert_keep(model, json.loads(Path(keep_path).expanduser().read_text()))
        print(f"tensorfold: TF_EXPERT_KEEP={keep_path}: router mask on {n} layers", flush=True)
    # the sparse prefill settings this process starts with (read again at every call), so a log shows the arm
    print(sparse_prefill.describe(), flush=True)
    eos = config.get("eos_token_id")
    tokenizer = load_tokenizer(Path(model_dir), eos_token_ids=eos if isinstance(eos, list) else None)
    return model, tokenizer
