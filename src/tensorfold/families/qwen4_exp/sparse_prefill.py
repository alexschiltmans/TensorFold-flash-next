"""Sparse prefill attention for Qwen3.8 Flash Next's QSA layers (TF_FLASH_SPARSE_PREFILL=1, default off).

Past 2,048 keys each query row of a sparse-attention layer reads only the 512 blocks (2,048 keys) its indexer
picks, plus its unfinished tail block. The reference prefill (``SparseAttention.__call__``) builds that choice as
a boolean mask over every key and runs MLX's dense attention under it: at 64K keys 32x the work the row needs,
and the larger part of a deep prompt chunk's time. Here a prompt call (more rows than a decode window, at least
one row past 512 complete blocks) instead:

1. picks each row's blocks with the dense path's own ops (``Indexer.prepare`` + the scores, -inf past the row's
   complete blocks, ``argpartition``), over the same row blocks ``_attend_in_blocks`` uses, so every row gets
   the same blocks the mask would give it;
2. turns them into key ids [rows, 4 * 512 + 3]: the blocks' keys in position order, then the tail keys
   [4 * complete, end) (a sparse row reads 2,048 + end - 4 * complete of them; a row with at most 512 complete
   blocks reads keys 0 .. end - 1, as causal attention does);
3. attends every row over its own keys in one kernel (``prefill_attention``: fp32 scores and softmax, keys
   read from the cache buffers by id, nothing gathered).

The masked-out keys contribute exactly zero to the dense path's softmax, so the two agree to rounding: the dense
path rounds the scores and probabilities to bf16, the kernel keeps them in fp32. Repeated calls are
bit-identical (no atomics; fixed orders).

Settings (read at every call; ``describe()`` gives them as one line, which ``model.load`` prints, and ``CALLS``
counts the calls ``attend`` took and declined in this process):
- TF_FLASH_SPARSE_PREFILL=1: on (``model.SparseAttention`` calls ``attend`` only then).
- TF_FLASH_SPARSE_KERNEL=grouped|rows: ``grouped`` (default) reads a row's keys once for the 12 query heads that
  share a KV head; ``rows`` is the decode kernel (a threadgroup per head and row).
- TF_FLASH_SPARSE_MIN_KEYS=n: calls ending before n keys keep the dense path (default 0).
- TF_FLASH_SPARSE_SELECT=exact|radix: how the top blocks are taken (``select_ids``; default exact).
- TF_FLASH_SPARSE_SELECT_ROWS=n: row block of the block choice (default: the dense path's blocks).
- TF_FLASH_SPARSE_ROWS=n: rows a kernel call takes (default 2048).
- TF_FLASH_SPARSE_PARTS=p: pieces each row's key list is cut into, merged in order (default 2 grouped, 1 rows).
"""

from __future__ import annotations

import math
import os
from typing import Any

import mlx.core as mx
import numpy as np

# calls of at most this many rows keep the dense path: decode steps and drafted windows (runtime.row_exact), whose
# rows must keep the bits a one-row step gives them
DECODE_ROWS = 16
KERNELS = ("grouped", "rows")
SELECTS = ("exact", "radix")

# calls ``attend`` got in this process: "sparse" it computed (and their "rows"), "declined" it left to the dense
# path (returned None). Read in-process; ``reset_calls()`` zeroes them.
CALLS = {"sparse": 0, "declined": 0, "rows": 0}


def reset_calls() -> dict[str, int]:
    """Zero ``CALLS``; returns the counts it had."""

    before = dict(CALLS)
    for key in CALLS:
        CALLS[key] = 0
    return before


def _env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def enabled() -> bool:
    return os.environ.get("TF_FLASH_SPARSE_PREFILL", "0").strip() not in ("", "0")


def settings() -> dict[str, Any]:
    """The effective settings now (unset ones at their defaults; ``select_rows`` 0: the dense path's blocks)."""

    kernel = os.environ.get("TF_FLASH_SPARSE_KERNEL", "grouped").strip() or "grouped"
    return {"on": enabled(), "min_keys": _env_int("TF_FLASH_SPARSE_MIN_KEYS", 0), "kernel": kernel,
            "select": os.environ.get("TF_FLASH_SPARSE_SELECT", "exact").strip() or "exact",
            "select_rows": _env_int("TF_FLASH_SPARSE_SELECT_ROWS"),
            "rows": max(1, _env_int("TF_FLASH_SPARSE_ROWS", 2048)),
            "parts": max(1, min(16, _env_int("TF_FLASH_SPARSE_PARTS", 2 if kernel == "grouped" else 1)))}


def describe() -> str:
    """One log line with the effective settings, e.g. ``[tensorfold] sparse prefill: on (min_keys 12288, kernel
    grouped, select exact, select_rows dense, rows 2048, parts 2)``. With the flag on, an unknown kernel or
    selection raises here (at load) rather than at the first deep prompt."""

    s = settings()
    if s["on"] and s["kernel"] not in KERNELS:
        raise ValueError(f"TF_FLASH_SPARSE_KERNEL={s['kernel']!r}: grouped or rows")
    if s["on"] and s["select"] not in SELECTS:
        raise ValueError(f"TF_FLASH_SPARSE_SELECT={s['select']!r}: exact or radix")
    return (f"[tensorfold] sparse prefill: {'on' if s['on'] else 'off'} (min_keys {s['min_keys']}, kernel "
            f"{s['kernel']}, select {s['select']}, select_rows {s['select_rows'] or 'dense'}, rows {s['rows']}, "
            f"parts {s['parts']})")


def _select_blocks(attn: Any, length: int, past: int) -> list[tuple[int, int]]:
    """Row blocks for the block choice: TF_FLASH_SPARSE_SELECT_ROWS rows if set, else the blocks the dense path
    chooses in (``_attend_in_blocks``' blocks, or the whole call where it selects at once), so each row's scores
    come from a matmul of the same shape; with no TF_FLASH_ATTN_MB / TF_FLASH_ATTN_ROWS, 256-row blocks."""

    from tensorfold.families.qwen4_exp.model import _FUSED_PREFILL_DIMS, _VECTOR_ROWS, _row_blocks

    step = _env_int("TF_FLASH_SPARSE_SELECT_ROWS")
    if step > 0:
        return _row_blocks(length, max(32, step), offset=past)
    block = attn._block_rows(past + length)
    if block > _VECTOR_ROWS and length > block and attn.dims not in _FUSED_PREFILL_DIMS:
        return _row_blocks(length, block, offset=past)
    if block > _VECTOR_ROWS:
        return [(0, length)]
    return _row_blocks(length, 256, offset=past)


def select_ids(attn: Any, index_query: mx.array, raw: mx.array, cache: Any, past: int, length: int,
               how: str | None = None) -> tuple[mx.array, np.ndarray, np.ndarray]:
    """Each row's key list: ids [length, 4 top + 3] int32 (valid up to counts[r] on sparse rows; unused on the
    others), counts [length] and sparse [length] (host arrays). Pools the blocks this call completes.

    ``how`` (TF_FLASH_SPARSE_SELECT): ``exact`` (default) takes the top blocks with the dense path's
    ``argpartition``; ``radix`` with the decode kernel's radix select (``kernels`` q4_idx_select) on the same
    scores, which breaks ties at the cut by the lowest block id, so a row with tied scores at its cut may read
    other blocks than the mask gives it."""

    how = how or os.environ.get("TF_FLASH_SPARSE_SELECT", "exact").strip() or "exact"
    if how not in ("exact", "radix"):
        raise ValueError(f"TF_FLASH_SPARSE_SELECT={how!r}: exact or radix")
    indexer = attn.indexer
    ratio, top = indexer.ratio, indexer.top_blocks
    width = ratio * top + ratio - 1
    ends = past + np.arange(length, dtype=np.int64) + 1
    complete = ends // ratio
    sparse = complete > top
    counts = np.where(sparse, ratio * top + ends - ratio * complete, ends).astype(np.int32)
    tail = mx.array(((complete * ratio)[:, None] + np.arange(ratio - 1, dtype=np.int64)[None, :]).astype(np.int32))

    q = indexer.prepare(index_query, raw, cache, past)          # float32 [B, HI, L, d], None if no row is sparse
    blocks = (past + length) // ratio
    pooled_t = cache.pooled[:, :blocks].astype(mx.float32)[:, None].transpose(0, 1, 3, 2)
    within = mx.arange(ratio, dtype=mx.int32)
    if how == "radix":
        from tensorfold.kernels.qwen.flash_next.v1 import kernels as K

        radix = K._kernel("q4_idx_select", K._IDX_SELECT, ["SC", "COMPLETE", "ENDS"], ["KEYS"],
                          header=K._SELECT_HEADER)
    pieces: list[mx.array] = []
    for start, stop in _select_blocks(attn, length, past):
        rows = stop - start
        if (past + stop) // ratio <= top:                         # every row of the block is causal
            pieces.append(mx.zeros((rows, width), dtype=mx.int32))
            continue
        # Indexer.select_rows' scores and top-k, op for op (the same shapes, so the same bits and the same blocks)
        row_ends = past + start + mx.arange(rows) + 1
        row_complete = row_ends // ratio
        scores = q[:, :, start:stop] @ pooled_t
        scores = mx.sum(mx.maximum(scores, 0), axis=1) / math.sqrt(indexer.dims)          # [B, rows, blocks]
        if how == "radix":                                         # reads only each row's complete blocks
            key_ids = radix(inputs=[scores[0], row_complete, row_ends], template=[("TOP", top), ("KW", width)],
                            grid=(1024 * rows, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(rows, width)],
                            output_dtypes=[mx.int32], init_value=0)[0]
        else:
            valid = mx.arange(blocks)[None, None, :] < row_complete[None, :, None]
            scores = mx.where(valid, scores, -mx.inf)
            chosen = mx.argpartition(scores, kth=-top, axis=-1)[..., -top:]
            chosen = mx.sort(chosen[0].astype(mx.int32), axis=-1)                          # position order
            key_ids = (chosen[:, :, None] * ratio + within).reshape(rows, ratio * top)
            key_ids = mx.concatenate([key_ids, tail[start:stop]], axis=1)
        mx.async_eval(key_ids)
        if pieces:
            mx.eval(pieces[-1])                                    # at most two blocks' scores alive
        pieces.append(key_ids)
    ids = pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=0)
    return ids, counts, sparse


def attend(attn: Any, queries: mx.array, index_query: mx.array, raw: mx.array, cache: Any, past: int
           ) -> mx.array | None:
    """Sparse prefill attention [B, H, L, D] for a call of ``queries`` [B, H, L, D] after ``past`` keys (its own
    keys already in ``cache``), or None where the dense path should run: a decode-sized call, no row past the
    indexer's budget, a call ending before TF_FLASH_SPARSE_MIN_KEYS keys, batch > 1, or not bf16."""

    batch, heads, length, dims = (int(x) for x in queries.shape)
    indexer = attn.indexer
    end = past + length
    s = settings()
    # every refusal comes before select_ids, so a declined call leaves the cache (pooled blocks) as it was
    if (batch != 1 or length <= DECODE_ROWS or end // indexer.ratio <= indexer.top_blocks or end < s["min_keys"]
            or queries.dtype != mx.bfloat16 or cache.keys.dtype != mx.bfloat16
            or cache.values.dtype != mx.bfloat16):
        CALLS["declined"] += 1
        return None
    ids, counts, sparse = select_ids(attn, index_query, raw, cache, past, length)
    kernel = s["kernel"]
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_attention as PA

    q_rows = queries[0].transpose(1, 0, 2)                         # [L, H, D]
    step = s["rows"]
    # a fixed part count, so a row's result does not depend on how many rows ride with it
    parts = s["parts"]
    outs: list[mx.array] = []
    for a in range(0, length, step):
        b = min(length, a + step)
        if kernel == "rows":
            out = PA.attention_rows(q_rows[a:b], cache.keys, cache.values, counts[a:b], ids[a:b], sparse[a:b],
                                    attn.scale, parts=parts)
        elif kernel == "grouped":
            out = PA.attention_grouped(q_rows[a:b], cache.keys, cache.values, mx.array(counts[a:b]), ids[a:b],
                                       mx.array(sparse[a:b].astype(np.int32)), attn.scale, parts=parts)
        else:
            raise ValueError(f"TF_FLASH_SPARSE_KERNEL={kernel!r}: grouped or rows")
        outs.append(out)
    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
    CALLS["sparse"] += 1
    CALLS["rows"] += length
    return out[None].transpose(0, 2, 1, 3)                         # [1, H, L, D]
