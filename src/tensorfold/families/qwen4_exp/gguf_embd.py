"""token_embd read row by row on the CPU (TF_TRUNK_EMBD_ROWS=1, default off), instead of a GPU-wrapped table.

With a ``trunk_source`` GGUF the embedding is mlx-kquant's ``KQuantEmbedding`` over a no-copy view of the file
(IQ4_XS in cut240, 0.32 GiB, counted in MLX active memory). Its call is ``weight[ids]``, then ``kq.dequantize`` of
those rows with the module's ``scales`` and codec, then a cast to bf16. Here the table is a numpy memmap of the same
bytes, never wrapped for the GPU: the call gathers the ids' rows on the CPU, makes an MLX array of them and runs the
very same ``kq.dequantize`` (the same 1-byte ``scales`` placeholder ``kq.load_gguf`` returns, the same reshape and
cast), so the output is bit-identical. The rows a call touches come through the page cache, outside MLX; a cold row is
a page fault.

Only the lookup is served: ``as_linear`` (a head tied to the embedding) raises, and the model loader refuses the
paths that read the table as a matrix or through their own kernels (fused decode, MTP and reference drafting).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class GGUFEmbedRows(nn.Module):
    """A ``KQuantEmbedding`` lookup over a memmap of the GGUF's rows; no parameters (nothing for ``load_weights``)."""

    def __init__(self, path: str | Path, offset: int, num_embeddings: int, dims: int, codec: str,
                 out_dtype: mx.Dtype = mx.bfloat16) -> None:
        from mlx_kquant.codec_geometry import bytes_per_row

        super().__init__()
        bpr = bytes_per_row(codec, dims)
        self.__dict__.update(num_embeddings=int(num_embeddings), dims=int(dims), kquant_type=codec,
                             out_dtype=out_dtype, row_bytes=int(bpr))
        self.__dict__["data"] = np.memmap(str(path), dtype=np.uint8, mode="r", offset=int(offset),
                                          shape=(int(num_embeddings), int(bpr)))
        # kq.load_gguf's ``.scales`` for a block codec: one zero byte. Evaluated here: a lazy array made at load keeps
        # the loading thread's stream (expert_stream._placeholder)
        scales = mx.zeros((1,), dtype=mx.uint8)
        mx.eval(scales)
        self.__dict__["scales_placeholder"] = scales

    @property
    def nbytes(self) -> int:
        return int(self.__dict__["data"].nbytes)

    def __call__(self, x: mx.array | np.ndarray) -> mx.array:
        import mlx_kquant as kq

        ids = np.asarray(x)
        flat = ids.reshape(-1).astype(np.int64)
        if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= self.num_embeddings):
            raise IndexError(f"token id out of range 0..{self.num_embeddings - 1}")
        rows = mx.array(self.__dict__["data"][flat])                       # [n, bytes_per_row] uint8, a copy
        deq = kq.dequantize(rows, self.__dict__["scales_placeholder"], self.kquant_type)
        return deq.reshape(*ids.shape, self.dims).astype(self.out_dtype)

    def as_linear(self, x: mx.array) -> mx.array:
        raise RuntimeError("TF_TRUNK_EMBD_ROWS=1 serves embedding lookups only: a head tied to the embedding needs "
                           "the GPU table (TF_TRUNK_EMBD_ROWS=0)")
