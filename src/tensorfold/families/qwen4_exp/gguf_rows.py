"""The PLE n-gram table read row by row from a GGUF file instead of held in memory.

llama.cpp's conversion stores the 128 checkpoint shards as one tensor, ``per_layer_token_embd.weight``
([rows, 160], the shards concatenated in order), so a flat row id indexes it directly. The file is memory-mapped
and only the rows a step needs are read and dequantized on the CPU (a decode step reads 16 rows of 90 bytes), so
the 28.8 GB table stays in the page cache, out of the GPU's working set.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

# ggml's kvalues_iq4nl: the 16 non-uniform levels of IQ4_NL
_IQ4_NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
_BLOCK, _BLOCK_BYTES = 32, 18          # 32 values: fp16 scale + 16 bytes of nibbles


class GGUFRows:
    def __init__(self, path: str, offset: int, rows: int, dims: int, qtype: str) -> None:
        if qtype != "IQ4_NL":
            raise ValueError(f"n-gram table type {qtype}: only IQ4_NL is implemented")
        if dims % _BLOCK:
            raise ValueError(f"row width {dims} is not a multiple of {_BLOCK}")
        self.rows, self.dims = rows, dims
        self.blocks = dims // _BLOCK
        self.data = np.memmap(path, dtype=np.uint8, mode="r", offset=offset,
                              shape=(rows, self.blocks * _BLOCK_BYTES))

    @classmethod
    def from_manifest(cls, model_dir: Path, manifest: str, dims: int) -> "GGUFRows":
        tables = json.loads((Path(model_dir) / manifest).read_text())
        if len(tables) != 1:
            raise ValueError(f"{manifest}: expected one table, found {len(tables)}")
        (entry,) = tables.values()
        path = Path(str(entry["file"])).expanduser()
        if not path.is_absolute():       # a self-contained checkpoint names the table file relative to itself
            path = Path(model_dir) / path
        return cls(str(path), int(entry["offset"]), int(entry["rows"]), dims, entry["type"])

    def numpy(self, ids: np.ndarray) -> np.ndarray:
        """Rows ``ids`` (any shape) -> float32 [..., dims]."""

        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.rows):
            raise IndexError("n-gram row id out of range")
        raw = self.data[flat].reshape(-1, self.blocks, _BLOCK_BYTES)
        scale = raw[..., :2].copy().view(np.float16).astype(np.float32)          # [n, blocks, 1]
        qs = raw[..., 2:]
        values = np.concatenate([_IQ4_NL[qs & 0xF], _IQ4_NL[qs >> 4]], axis=-1)  # low nibbles, then high
        return (values * scale).reshape(*np.shape(ids), self.dims)

    def __call__(self, ids: np.ndarray) -> mx.array:
        return mx.array(self.numpy(ids)).astype(mx.bfloat16)
