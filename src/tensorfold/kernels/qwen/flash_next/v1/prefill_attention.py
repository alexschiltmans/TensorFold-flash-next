"""Sparse attention for prompt rows (Qwen3.8 Flash Next, TF_FLASH_SPARSE_PREFILL).

Past 2,048 keys a sparse-attention row reads only its indexer's 512 blocks (2,048 keys) and its unfinished tail.
``kernels.attention_rows`` (the decode kernel) computes exactly that, a threadgroup per query head and row, so the
12 query heads that share a KV head each read the row's keys from memory. ``attention_grouped`` here is the same
arithmetic with the keys read once per row and KV head: a threadgroup per (KV head, row, part), its simdgroups the
group's query heads, the row's keys staged through threadgroup memory ``T`` at a time.

Per head: fp32 scores q . k (q pre-scaled; lane l holds dims l, l + 32, ...), an online softmax with one rescale
per tile, fp32 output, written as (max, sum, output) parts that ``kernels``' q4_attn_merge combines in part order.
A row's result depends on its own keys and ``parts`` only, not on the other rows.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.flash_next.v1 import kernels as K

_PREFILL_ATTN = r"""
  // Threadgroup (kvh, r, part), simdgroup s: query head kvh * G + s of row r over part `part` of the row's key
  // list (SPARSE[r]: the ids IDS[r][0 .. NK[r]); else keys 0 .. NK[r] - 1), entries [p n / P, (p + 1) n / P).
  constexpr int G = H / KVH;
  constexpr int PER = D / 32;
  constexpr int CH = D / 8;                      // 16-byte chunks a key row
  const uint lane = thread_index_in_simdgroup;
  const int s = int(simdgroup_index_in_threadgroup);
  const int t = int(thread_position_in_threadgroup.x);
  const int kvh = int(threadgroup_position_in_grid.x);
  const int r = int(threadgroup_position_in_grid.y);
  const int part = int(threadgroup_position_in_grid.z);
  const int h = kvh * G + s;
  const int n = NK[r];
  const int lo = int((long(part) * n) / P), hi = int((long(part + 1) * n) / P);
  const bool sparse = SPARSE[r] != 0;
  const size_t cap = size_t(Kc_shape[2]);
  const device bfloat* kb = Kc + size_t(kvh) * cap * D;
  const device bfloat* vb = Vc + size_t(kvh) * cap * D;
  const auto ids = IDS + size_t(r) * IDS_shape[1];
  threadgroup uint4 kt[T * CH];
  threadgroup uint4 vt[T * CH];
  const threadgroup bfloat* kh = reinterpret_cast<const threadgroup bfloat*>(kt);
  const threadgroup bfloat* vh = reinterpret_cast<const threadgroup bfloat*>(vt);
  const device bfloat* qp = Q + (size_t(r) * H + h) * D;
  float q[PER], o[PER];
  for (int i = 0; i < PER; i++) { q[i] = SCALE[0] * float(qp[lane + 32 * i]); o[i] = 0.0f; }
  float m = -INFINITY, l = 0.0f;
  for (int base = lo; base < hi; base += T) {
    const int cnt = min(T, hi - base);
    threadgroup_barrier(mem_flags::mem_threadgroup);           // the previous tile is consumed
    for (int c = t; c < cnt * CH; c += 32 * G) {
      const int j = c / CH, w = c % CH;
      const size_t key = size_t(sparse ? int(ids[base + j]) : base + j);
      kt[c] = reinterpret_cast<const device uint4*>(kb + key * D)[w];
      vt[c] = reinterpret_cast<const device uint4*>(vb + key * D)[w];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sc[T];
    float top = m;
    for (int j = 0; j < cnt; j++) {
      const threadgroup bfloat* kr = kh + j * D + lane;
      float d = 0.0f;
      for (int i = 0; i < PER; i++) d = fma(q[i], float(kr[32 * i]), d);
      sc[j] = simd_sum(d);
      top = metal::max(top, sc[j]);
    }
    const float f = metal::exp(m - top);
    l *= f;
    for (int i = 0; i < PER; i++) o[i] *= f;
    for (int j = 0; j < cnt; j++) {
      const float e = metal::exp(sc[j] - top);
      l += e;
      const threadgroup bfloat* vr = vh + j * D + lane;
      for (int i = 0; i < PER; i++) o[i] = fma(e, float(vr[32 * i]), o[i]);
    }
    m = top;
  }
  const size_t at = (size_t(r) * H + h) * P + part;
  for (int i = 0; i < PER; i++) PO[at * D + lane + 32 * i] = o[i];
  if (lane == 0) { PM[at * 2] = m; PM[at * 2 + 1] = l; }
"""

_scales: dict[float, mx.array] = {}


def attention_grouped(q: mx.array, keys: mx.array, values: mx.array, counts: mx.array, ids: mx.array,
                      sparse: mx.array, scale: float, *, parts: int = 1, tile: int = 16) -> mx.array:
    """Attention of each row's query heads q [R, H, D] (bf16) over its own keys, as ``kernels.attention_rows``:
    a sparse row (``sparse[r]`` != 0) the first counts[r] ids of ids[r] [R, W] (int32), a dense row keys
    0 .. counts[r] - 1, from the cache buffers keys / values [1, KVH, cap, D] (bf16). [R, H, D] bf16."""

    rows, heads, dims = (int(x) for x in q.shape)
    kv_heads = int(keys.shape[1])
    if heads % kv_heads or dims % 32 or (heads // kv_heads) * 32 > 1024:
        raise ValueError(f"attention_grouped: {heads} heads over {kv_heads} KV heads of {dims} dims")
    group = heads // kv_heads
    scale_arr = _scales.get(scale)
    if scale_arr is None:
        scale_arr = _scales[scale] = mx.array([scale], dtype=mx.float32)
    kernel = K._kernel("q4_prefill_attn", _PREFILL_ATTN, ["Q", "Kc", "Vc", "IDS", "NK", "SPARSE", "SCALE"],
                       ["PO", "PM"])
    po, pm = kernel(inputs=[q, keys, values, ids, counts, sparse, scale_arr],
                    template=[("H", heads), ("KVH", kv_heads), ("D", dims), ("P", parts), ("T", tile)],
                    grid=(32 * group * kv_heads, rows, parts), threadgroup=(32 * group, 1, 1),
                    output_shapes=[(rows, heads, parts, dims), (rows, heads, parts, 2)],
                    output_dtypes=[mx.float32, mx.float32])
    merge = K._kernel("q4_attn_merge", K._ATTN_MERGE, ["PO", "PM"], ["OUT"])
    return merge(inputs=[po, pm], template=[("H", heads), ("D", dims), ("P", parts)],
                 grid=(dims, heads, rows), threadgroup=(dims, 1, 1),
                 output_shapes=[(rows, heads, dims)], output_dtypes=[mx.bfloat16])[0]


def attention_rows(q: mx.array, keys: mx.array, values: mx.array, counts: Any, ids: mx.array, sparse: Any,
                   scale: float, *, parts: int = 1) -> mx.array:
    """The decode kernel (``kernels.attention_rows``, a threadgroup per query head and row) on prompt rows."""

    return K.attention_rows(q, keys, values, [int(c) for c in counts], ids, [bool(x) for x in sparse], scale,
                            parts=parts)
