"""Routed experts read in place from a llama.cpp GGUF file (mlx-kquant), instead of the checkpoint's MLX arrays.

A checkpoint whose config.json names an ``experts_source`` keeps its own trunk (attention, DeltaNet, shared experts,
routers, hyper-connections, PLE, embeddings, head) and takes every layer's routed experts
(``mlp.switch_mlp.{gate,up,down}_proj``) from the GGUF instead::

    "experts_source": {"format": "gguf-kquant", "file": "/path/to/model-00001-of-00002.gguf"}

- i-quant expert stacks (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL) become ``mlx_kquant.nn.KQuantSwitchLinear``
  modules whose ``weight`` is the GGUF's own bytes: ``kq.load_gguf(zero_copy=True)`` wraps a page-aligned window of
  its read-only mmap in a no-copy Metal buffer, so the experts stay clean file-backed pages (the page cache, as
  llama.cpp's mmap), not anonymous MLX memory. A GGUF ``blk.L.ffn_{gate,up,down}_exps.weight`` [K, N, E] is
  already kquant's (E, N, row_bytes) layout; GGUF expert i is router row i.
- Q2_0 expert stacks (``(q - 1) * d``, fp16 d; kquant has no Q2_0 codec) are repacked once into MLX affine 2-bit
  groups of 64: the 2-bit codes copy verbatim, scales d and biases -d stay fp16 (bf16 scales would round the
  weights). These are anonymous memory (~131 MB a layer). Their activations go in as fp16 (TF_FLASH_Q2_X=f16, the
  default; bf16 activations against fp16 scales take MLX's mixed-dtype path, float32 out, 4.3x slower at one row
  and 1.24x at 512 sorted rows in FM3k's bench), and the output is cast back to x's dtype. TF_FLASH_Q2_X=bf16
  keeps the float32 path.
- Decode (fewer than 64 routed rows, SwitchGLU's unsorted gather): gate and up run as one kquant
  ``moe_glu_gather_kq`` dispatch (silu(g) * u fused; FM3k: about half the time of two gathers) when both share a
  codec that has the fused kernel. TF_FLASH_KQ_GLU=0 turns it off. Prefill (sorted rows) takes SwitchGLU's path.

The checkpoint's expert arrays are dropped from ``mx.load``'s lazy dict before anything evaluates them, so they are
never read. Nothing here wires memory: no ``mx.set_wired_limit`` and none of kquant's arena or residency calls.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

FORMAT = "gguf-kquant"
_PROJ = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
# ggml type id -> (name, kquant codec, weights per block, bytes per block); Q2_0 has no codec (affine repack)
_TYPES = {
    18: ("IQ3_XXS", "iq3_xxs", 256, 98),
    20: ("IQ4_NL", "iq4_nl", 32, 18),
    21: ("IQ3_S", "iq3_s", 256, 110),
    22: ("IQ2_S", "iq2_s", 256, 82),
    23: ("IQ4_XS", "iq4_xs", 256, 136),
    42: ("Q2_0", None, 64, 18),
}
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


@dataclass
class _Tensor:
    name: str
    dims: tuple[int, ...]   # GGUF order, innermost first: an expert stack is (K, N, E)
    ggml_type: int
    offset: int             # absolute file offset of the data


def read_gguf_tensors(path: Path) -> dict[str, _Tensor]:
    """The tensor table of a GGUF file (names, dims, types, absolute data offsets); reads the header only."""

    with open(path, "rb", buffering=1 << 20) as f:
        def read(n: int) -> bytes:
            b = f.read(n)
            if len(b) != n:
                raise EOFError(f"{path}: short read in the GGUF header")
            return b

        def u32() -> int:
            return struct.unpack("<I", read(4))[0]

        def u64() -> int:
            return struct.unpack("<Q", read(8))[0]

        def skip_value(vtype: int) -> Any:
            if vtype == 8:
                return read(u64()).decode("utf-8", errors="replace")
            if vtype == 9:
                itype, count = u32(), u64()
                if itype == 8:
                    for _ in range(count):
                        read(u64())
                    return None
                read(struct.calcsize(_SCALAR[itype]) * count)
                return None
            fmt = _SCALAR[vtype]
            return struct.unpack(fmt, read(struct.calcsize(fmt)))[0]

        if read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        u32()
        n_tensors, n_kv = u64(), u64()
        align = 32
        for _ in range(n_kv):
            key = read(u64()).decode("utf-8", errors="replace")
            value = skip_value(u32())
            if key == "general.alignment":
                align = int(value)
        infos = []
        for _ in range(n_tensors):
            name = read(u64()).decode("utf-8")
            dims = tuple(u64() for _ in range(u32()))
            infos.append((name, dims, u32(), u64()))
        start = (f.tell() + align - 1) // align * align
    return {n: _Tensor(n, d, t, start + rel) for n, d, t, rel in infos}


class AffineQ2SwitchLinear(nn.Module):
    """GGUF Q2_0 expert stacks as MLX affine 2-bit, groups of 64, fp16 scales and biases (exact)."""

    def __init__(self, num_experts: int, output_dims: int, input_dims: int) -> None:
        super().__init__()
        self.group_size, self.bits, self.mode = 64, 2, "affine"
        self.weight = mx.zeros((num_experts, output_dims, input_dims // 16), dtype=mx.uint32)
        self.scales = mx.zeros((num_experts, output_dims, input_dims // 64), dtype=mx.float16)
        self.biases = mx.zeros((num_experts, output_dims, input_dims // 64), dtype=mx.float16)
        self.freeze()

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        xin = x if os.environ.get("TF_FLASH_Q2_X", "f16") == "bf16" else x.astype(mx.float16)
        y = mx.gather_qmm(xin, self["weight"], self["scales"], self["biases"], rhs_indices=indices, transpose=True,
                          group_size=64, bits=2, sorted_indices=sorted_indices)
        return y.astype(x.dtype)


def _switch_glu_class() -> type:
    from mlx_lm.models.switch_layers import SwiGLU, SwitchGLU

    class GGUFSwitchGLU(SwitchGLU):
        """mlx-lm's SwitchGLU over GGUF expert modules (same parameter paths), with kquant's fused gate/up GLU
        for decode-sized calls."""

        def __init__(self, gate: nn.Module, up: nn.Module, down: nn.Module) -> None:
            nn.Module.__init__(self)
            self.gate_proj, self.up_proj, self.down_proj = gate, up, down
            self.activation = SwiGLU()
            codec = getattr(gate, "kquant_type", None)
            import mlx_kquant as kq

            fusable = codec is not None and codec == getattr(up, "kquant_type", None) and kq.codec_has_moe_glu(codec)
            self.__dict__["fused_codec"] = codec if fusable else None

        def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
            codec = self.__dict__.get("fused_codec")
            if codec is None or indices.size >= 64 or os.environ.get("TF_FLASH_KQ_GLU", "1") == "0":
                return super().__call__(x, indices)
            import mlx_kquant as kq

            lead, top = indices.shape[:-1], indices.shape[-1]
            idx = indices.reshape(-1, top)
            h = kq.moe_glu_gather_kq(x.reshape(-1, x.shape[-1]), self.gate_proj["weight"], self.up_proj["weight"],
                                     codec, idx)                                   # [T, R, N]
            y = self.down_proj(h[:, :, None, :], idx, sorted_indices=False)          # [T, R, 1, D]
            return y.reshape(*lead, top, y.shape[-1])

    return GGUFSwitchGLU


def q2_0_to_affine(raw: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q2_0 rows (..., k/64 * 18 bytes: fp16 d, then 16 bytes of codes) -> MLX affine 2-bit g64 parts:
    codes uint32 (..., k/16), scales d and biases -d, fp16 (..., k/64). ggml puts element j at byte j/4, bits
    2*(j%4); MLX reads element i of a uint32 at bits 2*(i%16), little-endian: the same bits."""

    lead = raw.shape[:-1]
    blocks = raw.reshape(*lead, k // 64, 18)
    d = np.ascontiguousarray(blocks[..., :2]).view(np.float16)[..., 0]
    codes = np.ascontiguousarray(blocks[..., 2:]).reshape(*lead, k // 4).view(np.uint32)
    return codes, d.copy(), (-d).astype(np.float16)


class GGUFExperts:
    """Plan, modules and arrays for one model's routed experts from a GGUF file."""

    def __init__(self, model_dir: Path, source: dict[str, Any], num_layers: int, num_experts: int,
                 hidden: int, intermediate: int, counts: list[int] | None = None) -> None:
        # counts: per-layer expert counts of a cut file (experts_cut, model.load): layer L's stacks hold
        # counts[L] experts (storage order = ascending router ids, see model.set_expert_cut); None = num_experts
        if source.get("format") != FORMAT:
            raise ValueError(f"experts_source format {source.get('format')!r}: only {FORMAT!r} is implemented")
        path = Path(str(source["file"])).expanduser()
        if not path.is_absolute():
            path = Path(model_dir) / path
        self.path = path.resolve()
        self.require_zero_copy = bool(source.get("require_zero_copy", True))
        tensors = read_gguf_tensors(self.path)
        self.all_names = list(tensors)
        self.plan: dict[tuple[int, str], _Tensor] = {}
        for layer in range(num_layers):
            for proj, short in _PROJ.items():
                name = f"blk.{layer}.ffn_{short}_exps.weight"
                t = tensors.get(name)
                if t is None:
                    raise ValueError(f"{self.path.name}: no {name}")
                k, n = (hidden, intermediate) if short != "down" else (intermediate, hidden)
                e = counts[layer] if counts is not None else num_experts
                if t.dims != (k, n, e):
                    raise ValueError(f"{name}: dims {t.dims}, expected {(k, n, e)}")
                if t.ggml_type not in _TYPES:
                    raise ValueError(f"{name}: ggml type {t.ggml_type} is not handled here")
                self.plan[(layer, proj)] = t
        self.num_experts, self.hidden, self.intermediate = num_experts, hidden, intermediate
        self.counts = list(counts) if counts is not None else None

    def codec(self, layer: int, proj: str) -> str | None:
        return _TYPES[self.plan[(layer, proj)].ggml_type][1]

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for t in self.plan.values():
            kinds[_TYPES[t.ggml_type][0]] = kinds.get(_TYPES[t.ggml_type][0], 0) + 1
        return {"file": str(self.path), "tensors": kinds, "zero_copy_bytes": self.file_bytes(),
                "repacked_bytes": self.repacked_bytes()}

    def _nbytes(self, t: _Tensor) -> int:
        _, _, wpb, bpb = _TYPES[t.ggml_type]
        return t.dims[0] // wpb * bpb * t.dims[1] * t.dims[2]

    def file_bytes(self) -> int:
        """Bytes of the expert stacks used in place (file-backed)."""

        return sum(self._nbytes(t) for t in self.plan.values() if _TYPES[t.ggml_type][1])

    def repacked_bytes(self) -> int:
        """Bytes of the Q2_0 stacks' affine repack (anonymous): 2-bit codes plus fp16 scales and biases."""

        total = 0
        for t in self.plan.values():
            if _TYPES[t.ggml_type][1] is None:
                k, n, e = t.dims
                total += e * n * (k // 4 + 2 * 2 * (k // 64))
        return total

    def ranges(self) -> list[tuple[str, int, int]]:
        """(name, file offset, bytes) of every expert stack used in place: what a residency probe maps."""

        return [(t.name, t.offset, self._nbytes(t)) for t in self.plan.values() if _TYPES[t.ggml_type][1]]

    def install(self, model: nn.Module) -> None:
        """Swap each layer's SwitchGLU projections for kquant (or affine Q2_0) modules, before load_weights."""

        from mlx_kquant.nn import KQuantSwitchLinear

        glu = _switch_glu_class()
        for layer_index, layer in enumerate(model.layers):
            switch = layer.mlp.switch_mlp
            made = {}
            for proj in _PROJ:
                old = getattr(switch, proj)
                experts, out_dims, in_dims = old.weight.shape
                if self.counts is not None:
                    experts = self.counts[layer_index]          # a cut file's stack (load_weights checks shapes)
                codec = self.codec(layer_index, proj)
                made[proj] = (KQuantSwitchLinear(experts, out_dims, in_dims, False, codec) if codec
                              else AffineQ2SwitchLinear(experts, out_dims, in_dims))
            layer.mlp.switch_mlp = glu(made["gate_proj"], made["up_proj"], made["down_proj"])

    def arrays(self) -> dict[str, mx.array]:
        """The expert parameters by module path (``model.layers.L.mlp.switch_mlp.P.{weight,scales[,biases]}``).

        Raises if a stack meant to be used in place came back as a copy (``require_zero_copy``, the default): a
        copy would be anonymous memory, which is what this route exists to avoid."""

        import mlx_kquant as kq

        wanted = {t.name: key for key, t in self.plan.items() if _TYPES[t.ggml_type][1]}
        before = kq.zero_copy_view_count()
        arrays, codecs, _meta, _shapes = kq.load_gguf(str(self.path), zero_copy=True,
                                                      skip=[n for n in self.all_names if n not in wanted])
        codecs = dict(codecs)
        views = kq.zero_copy_view_count() - before
        if self.require_zero_copy and views != len(wanted):
            raise RuntimeError(f"kq.load_gguf wrapped {views} of {len(wanted)} expert stacks without a copy; the rest "
                               "would be anonymous memory (experts_source.require_zero_copy)")
        out: dict[str, mx.array] = {}
        for name, (layer, proj) in wanted.items():
            codec = self.codec(layer, proj)
            if codecs.get(name) != codec:
                raise ValueError(f"{name}: kquant reads codec {codecs.get(name)!r}, expected {codec!r}")
            prefix = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
            out[f"{prefix}.weight"] = arrays[name]
            out[f"{prefix}.scales"] = arrays[name[: -len(".weight")] + ".scales"]
        extra = set(arrays) - set(wanted) - {n[: -len(".weight")] + ".scales" for n in wanted}
        if extra:
            raise ValueError(f"kq.load_gguf returned tensors that were meant to be skipped: {sorted(extra)[:4]}")
        for (layer, proj), t in self.plan.items():
            if _TYPES[t.ggml_type][1]:
                continue
            k, n, e = t.dims
            raw = np.memmap(self.path, dtype=np.uint8, mode="r", offset=t.offset, shape=(e, n, k // 64 * 18))
            codes, scales, biases = q2_0_to_affine(raw, k)
            del raw
            prefix = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
            out[f"{prefix}.weight"] = mx.array(codes)
            out[f"{prefix}.scales"] = mx.array(scales)
            out[f"{prefix}.biases"] = mx.array(biases)
        return out


def is_expert_key(key: str) -> bool:
    """A checkpoint key (after sanitize) that belongs to the routed experts, which experts_source replaces."""

    return ".mlp.switch_mlp." in key
