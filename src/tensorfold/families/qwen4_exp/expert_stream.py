"""Routed experts streamed from the GGUF into a capped resident arena (TF_EXPERT_STREAM=1, default off).

With an ``experts_source`` GGUF, the stock loader maps every layer's whole expert stack for the GPU. Here the file is
never mapped for the GPU: each layer keeps C_L = min(C, E_L) expert slots in ``kq.arena_alloc`` buffers (host pages
wrapped no-copy as Metal shared buffers, in exactly the layout the stock modules hold), and every routed expert still
runs. A routed expert that is not resident (a miss) is read from the file into a slot before the layer's gather is
built, so outputs equal the all-resident build bit for bit: the same bytes, the same kernels.

- **Arena path** (calls with fewer than 64 routed ids, SwitchGLU's unsorted gather: decode and short continuations):
  one host sync on the layer's routed ids, an exclusive LRU over the slots (a miss takes the least recently used
  slot whose expert this call does not route to; empty slots first), the call's misses read together, then the
  stock ``GGUFSwitchGLU`` over the arena with the ids remapped to slots. Sorted calls never take it: the sorted
  kernels choose their variant and tile map from the stack's expert count, which a compact arena would change.
- **Ring path** (every other call: prompt chunks): one ring per projection and part, sized for the largest layer
  and viewed per layer as that layer's full stack [E_L, ...]. Only the call's routed experts are staged into their
  own positions (resident ones copied from the arena, the rest read from the file); unrouted positions keep stale
  bytes no kernel reads. The stock call then runs with the unchanged ids on a stack of the stock shape. The arena is
  not evicted by prefill (hits refresh their recency); after a prompt's final chunk the experts its last
  TF_EXPERT_WARM_ROWS rows routed are copied ring -> arena, no I/O.
- **Shared ring** (TF_EXPERT_RING=shared; default split, the ring above): one buffer per (codec family, part), sized
  for the largest single projection stack and viewed per (layer, projection) as that projection's [E_L, ...] stack,
  so a ring call holds one projection at a time. It runs mlx-lm ``SwitchGLU``'s ops in their order (``_gather_sort``,
  up, gate, activation, down, ``_scatter_unsort``) with a stage before each projection and an eval after up and after
  gate, so a stage never overwrites bytes a queued kernel still reads; down is fenced by ``done`` as before. Warming
  copies each projection ring -> arena while it is staged (slots assigned once, before the first stage). Unsorted ring
  calls (fewer than 64 routed ids, more distinct experts than slots) with kquant's fused gate/up GLU need gate and up
  at once: gate goes to the shared buffer and up to a small pair buffer (SORT_MIN - 1 experts), both as compact
  stacks with the ids remapped, as the arena path does; then down, compact, after the GLU's eval.
- **Reads**: one ``O_RDONLY`` descriptor with ``F_NOCACHE``, ``os.preadv`` from a thread pool. A job is read as its
  16 KiB-aligned superset into a per-thread page-aligned scratch buffer, then the exact slice is copied out, so the
  reads leave no partial pages in the page cache; adjacent jobs coalesce into one run, runs split into pieces of at
  most 4 MiB. Q2_0 stacks are read into host scratch, repacked to MLX affine 2-bit (``q2_0_to_affine``) in full,
  then copied into the slot. A short read or an error raises after the slots it was filling are marked empty.
- **Forks**: every arena and ring buffer is ``VM_INHERIT_NONE`` (absent from a forked child). Pages a Metal buffer
  has used cannot be shared copy-on-write, so without it every ``fork`` (any ``subprocess`` call of this process)
  copied the whole arena into the child: ~74 ms a GiB, at 10 GiB ~0.7 s a fork and ~11 GB of transient memory.
- **Pinning** (TF_EXPERT_PIN=1): every arena and ring buffer is ``mlock``ed and added to kquant's standalone
  residency set (committed, residency requested), both released on ``close``. Unpinned, the pages are wired only
  while a command buffer uses them and macOS compresses or swaps them in between, which at a large arena turns into
  constant swapping. Pinned bytes (arena + ring) are capped at ``pin_cap()``: PIN_FRACTION of the machine's memory,
  at most PIN_MAX_GIB. When the pin does not fit under the cap, or ``mlock`` (vm.user_wire_limit) or kquant's
  residency set refuses it, the stream logs why and runs unpinned instead of failing the load.
- **The fence**: a slot or the ring is overwritten only after the GPU has finished every earlier reader. Each MoE
  hands its output to ``done()``; every layer call evaluates the previous one right after its router sync. Within a
  forward that output is an ancestor of the router (free); at layer 0 it is the previous forward's last MoE output,
  which every gather of that forward feeds. One sequence at a time: a batch above 1 and interleaved forwards raise.
- **Resident layers** (TF_EXPERT_RESIDENT=L,L,...): the arena holds all E_L experts in expert order (slot i is expert
  i), read once at load with the same reads and repack, pinned with the rest, and never written again. The layer's
  call is the stock one, ``switch(x, experts)`` with the router's ids as they are: no host sync, no plan, no ring, no
  fence (its buffers are read-only, so no reader needs one) and no stats record. It still takes its turn in the
  order and thread checks. Its ``done`` is a no-op, so the fence stays the last streamed layer's output, which every
  streamed gather of its forward feeds. TF_EXPERT_LAYER_SLOTS=L:C,... gives single streamed layers their own C.

Settings (``settings()``, read once at load): TF_EXPERT_STREAM=1; TF_EXPERT_ARENA_SLOTS=C or TF_EXPERT_ARENA_GIB=g
(the largest equal C whose arena fits g GiB); TF_EXPERT_IO_THREADS (8); TF_EXPERT_WARM_ROWS (64, 0 off);
TF_EXPERT_PIN=1 (pin the arena and ring, above); TF_EXPERT_RING=split|shared (above); TF_EXPERT_RESIDENT and
TF_EXPERT_LAYER_SLOTS (resident layers and per-layer slots, above; with TF_EXPERT_ARENA_GIB the budget left after
both sets the equal C of the other layers); TF_EXPERT_PREFILL_ROWS (2048: a ring call with fewer rows is a prompt's final chunk, unless the caller says so with
``ExpertStream.final_chunk``); TF_EXPERT_MINCORE=1 (label each read cold or cached by ``mincore`` on a separate
read-only mapping, never mapped for the GPU); TF_EXPERT_STATS=path.jsonl (one line per call; without it only the last CALLS_KEPT calls are kept in memory);
TF_EXPERT_DEBUG=1 (check the modules' buffers against the arena and ring on every call; always checked once at load).
"""

from __future__ import annotations

import ctypes
import fcntl
import json
import mmap
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

F_NOCACHE = 48
ALIGN = 16384                 # read granularity: offsets round down and ends up to this
PIECE = 4 << 20               # the longest single read (a coalesced run splits into pieces)
SORT_MIN = 64                 # mlx-lm SwitchGLU sorts rows from this many routed ids (``indices.size >= 64``)
MIN_SLOTS = 16
PIN_MAX_GIB = 21.0            # the most bytes pinned (arena + ring), on any machine
PIN_FRACTION = 0.4375        # ... and at most this share of the machine's memory (21 GiB of 48 GiB)
CALLS_KEPT = 4096             # per-call records kept in memory between drains
VM_INHERIT_NONE = 2
PAGE = os.sysconf("SC_PAGE_SIZE")
_PROJS = ("gate_proj", "up_proj", "down_proj")


def _env_flag(env: Any, name: str) -> bool:
    return str(env.get(name, "0")).strip() not in ("", "0")


def settings(env: Any = None) -> dict[str, Any] | None:
    """The TF_EXPERT_* switches, validated; None when streaming is off (TF_EXPERT_STREAM unset or 0)."""

    env = os.environ if env is None else env
    on = str(env.get("TF_EXPERT_STREAM", "0")).strip()
    if on in ("", "0"):
        return None
    if on != "1":
        raise ValueError(f"TF_EXPERT_STREAM={on!r}: 1 or 0")
    slots, gib = str(env.get("TF_EXPERT_ARENA_SLOTS", "")).strip(), str(env.get("TF_EXPERT_ARENA_GIB", "")).strip()
    if bool(slots) == bool(gib):
        raise ValueError("TF_EXPERT_STREAM=1 needs exactly one of TF_EXPERT_ARENA_SLOTS and TF_EXPERT_ARENA_GIB")
    out: dict[str, Any] = {"slots": int(slots) if slots else None, "gib": float(gib) if gib else None}
    out["threads"] = int(env.get("TF_EXPERT_IO_THREADS", "8") or 8)
    out["warm_rows"] = int(env.get("TF_EXPERT_WARM_ROWS", "64") or 0)
    out["prefill_rows"] = int(env.get("TF_EXPERT_PREFILL_ROWS", "2048") or 2048)
    out["mincore"] = _env_flag(env, "TF_EXPERT_MINCORE")
    out["pin"] = _env_flag(env, "TF_EXPERT_PIN")
    out["stats"] = str(env.get("TF_EXPERT_STATS", "")).strip() or None
    out["debug"] = _env_flag(env, "TF_EXPERT_DEBUG")
    out["ring"] = str(env.get("TF_EXPERT_RING", "split")).strip() or "split"
    if out["ring"] not in ("split", "shared"):
        raise ValueError(f"TF_EXPERT_RING={out['ring']!r}: split or shared")
    if not 1 <= out["threads"] <= 64 or out["warm_rows"] < 0 or out["prefill_rows"] < 1:
        raise ValueError(f"TF_EXPERT_*: threads 1..64, warm rows >= 0, prefill rows >= 1 ({out})")
    if out["gib"] is not None and out["gib"] <= 0:
        raise ValueError(f"TF_EXPERT_ARENA_GIB={gib}: must be positive")
    out["resident"] = _layer_list(str(env.get("TF_EXPERT_RESIDENT", "")))
    out["layer_slots"] = _layer_slots(str(env.get("TF_EXPERT_LAYER_SLOTS", "")))
    both = sorted(set(out["resident"]) & set(out["layer_slots"]))
    if both:
        raise ValueError(f"TF_EXPERT_RESIDENT and TF_EXPERT_LAYER_SLOTS both name layers {both}: a resident layer "
                         "holds all its experts")
    return out


def _layer_index(text: str, name: str) -> int:
    try:
        i = int(text)
    except ValueError:
        raise ValueError(f"{name}: {text!r} is not a layer index") from None
    if i < 0:
        raise ValueError(f"{name}: layer {i} is negative")
    return i


def _layer_list(text: str) -> list[int]:
    """TF_EXPERT_RESIDENT: comma-separated model layer indices, each once (empty: none)."""

    items = [t.strip() for t in text.split(",")] if text.strip() else []
    layers = [_layer_index(t, "TF_EXPERT_RESIDENT") for t in items]
    if len(set(layers)) != len(layers):
        raise ValueError(f"TF_EXPERT_RESIDENT={text!r}: a layer is named twice")
    return sorted(layers)


def _layer_slots(text: str) -> dict[int, int]:
    """TF_EXPERT_LAYER_SLOTS: comma-separated ``layer:slots``, each layer once, slots at least MIN_SLOTS."""

    out: dict[int, int] = {}
    for item in [t.strip() for t in text.split(",")] if text.strip() else []:
        layer, sep, slots = item.partition(":")
        if not sep:
            raise ValueError(f"TF_EXPERT_LAYER_SLOTS: {item!r} is not layer:slots")
        i = _layer_index(layer.strip(), "TF_EXPERT_LAYER_SLOTS")
        try:
            c = int(slots)
        except ValueError:
            raise ValueError(f"TF_EXPERT_LAYER_SLOTS: {item!r}: slots must be an integer") from None
        if i in out:
            raise ValueError(f"TF_EXPERT_LAYER_SLOTS: layer {i} is named twice")
        if c < MIN_SLOTS:
            raise ValueError(f"TF_EXPERT_LAYER_SLOTS: layer {i}: {c} slots, at least {MIN_SLOTS}")
        out[i] = c
    return out


def refuse(cfg: dict[str, Any], config: dict[str, Any], env: Any = None) -> None:
    """Combinations streaming does not serve, each a ValueError with the reason (checked before anything loads)."""

    env = os.environ if env is None else env
    if not config.get("experts_source"):
        raise ValueError("TF_EXPERT_STREAM=1 needs experts_source (a GGUF expert file to stream from)")
    wired = str(env.get("TF_WIRED_GIB", "")).strip()
    if wired not in ("", "0", "0.0"):
        raise ValueError("TF_EXPERT_STREAM=1 with TF_WIRED_GIB: the arena is wired by its own residency set; a wired "
                         "limit on top of it is not supported")
    draft = str(env.get("TF_FLASH_REF_DRAFT", "off")).strip().lower() or "off"
    if draft != "off":
        raise ValueError(f"TF_EXPERT_STREAM=1 with TF_FLASH_REF_DRAFT={draft}: drafting is not served with streaming")
    if str(env.get("TF_FLASH_MTP_GGUF", "")).strip():
        raise ValueError("TF_EXPERT_STREAM=1 with TF_FLASH_MTP_GGUF: MTP drafting is not served with streaming")
    if cfg.get("slots") is not None and cfg["slots"] < MIN_SLOTS:
        raise ValueError(f"TF_EXPERT_ARENA_SLOTS={cfg['slots']}: at least {MIN_SLOTS}")


# -- plans ------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjPlan:
    """One layer's expert stack of one projection: where it sits in the file and how a slot holds an expert."""

    layer: int
    proj: str
    codec: str | None        # kquant codec; None = Q2_0 (affine 2-bit repack)
    offset: int              # file offset of expert 0
    experts: int             # E_L, the stack's expert count (storage order)
    rows: int                # N, output rows of an expert
    k: int                   # input dims
    file_bytes: int          # one expert's bytes in the file (contiguous: expert e at offset + e * file_bytes)

    def parts(self) -> list[tuple[str, int, tuple[int, ...], Any]]:
        """(name, itemsize, per-expert shape, mx dtype) of what a slot holds, in the stock module's layout."""

        if self.codec is not None:
            return [("weight", 1, (self.rows, self.file_bytes // self.rows), mx.uint8)]
        return [("weight", 4, (self.rows, self.k // 16), mx.uint32), ("scales", 2, (self.rows, self.k // 64), mx.float16),
                ("biases", 2, (self.rows, self.k // 64), mx.float16)]

    @property
    def served_bytes(self) -> int:
        return sum(int(np.prod(shape)) * size for _, size, shape, _ in self.parts())


def plans_from(experts: Any) -> list[dict[str, ProjPlan]]:
    """Per layer, per projection, the ``ProjPlan`` of a ``gguf_experts.GGUFExperts``."""

    from tensorfold.families.qwen4_exp.gguf_experts import _TYPES

    layers = sorted({layer for layer, _ in experts.plan})
    out = []
    for layer in layers:
        row = {}
        for proj in _PROJS:
            t = experts.plan[(layer, proj)]
            k, n, e = t.dims
            _, codec, wpb, bpb = _TYPES[t.ggml_type]
            row[proj] = ProjPlan(layer, proj, codec, t.offset, e, n, k, k // wpb * bpb * n)
        out.append(row)
    return out


def arena_bytes(plans: list[dict[str, ProjPlan]], slots: int) -> int:
    return sum(min(slots, p["gate_proj"].experts) * sum(q.served_bytes for q in p.values()) for p in plans)


def slots_for_gib(plans: list[dict[str, ProjPlan]], gib: float, fixed: int = 0) -> int:
    """The largest equal slot count whose arena fits ``gib`` GiB (less ``fixed`` bytes spent on other layers)."""

    budget, top = gib * 2**30 - fixed, max(p["gate_proj"].experts for p in plans)
    best = 0
    for c in range(1, top + 1):
        if arena_bytes(plans, c) > budget:
            break
        best = c
    return best


# -- reads ------------------------------------------------------------------------
_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
_libc.mlock.argtypes = _libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.minherit.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


class ExpertReader:
    """Exact byte ranges of one file, read with F_NOCACHE as aligned supersets, optionally labelled by mincore."""

    def __init__(self, path: str | Path, threads: int = 8, mincore: bool = False) -> None:
        self.path = str(path)
        self.fd = os.open(self.path, os.O_RDONLY)
        fcntl.fcntl(self.fd, F_NOCACHE, 1)
        self.size = os.fstat(self.fd).st_size
        self.threads = int(threads)
        self.pool = ThreadPoolExecutor(self.threads, thread_name_prefix="expert-io") if self.threads > 1 else None
        self._local = threading.local()
        self._base = None
        if mincore:
            base = _libc.mmap(None, self.size, mmap.PROT_READ, mmap.MAP_SHARED, self.fd, 0)
            if base in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), f"{self.path}: mmap for mincore failed")
            self._base = base

    @property
    def mincore(self) -> bool:
        return self._base is not None

    def residency(self, off: int, length: int) -> int:
        """Bytes of [off, off + length) whose pages are resident in RAM (page granularity, clipped to the range)."""

        if self._base is None or length <= 0:
            return 0
        a = off // PAGE * PAGE
        n = (off + length - a + PAGE - 1) // PAGE
        vec = ctypes.create_string_buffer(n)
        if _libc.mincore(ctypes.c_void_p(self._base + a), n * PAGE, vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore failed")
        flags = np.frombuffer(vec.raw, dtype=np.uint8) & 1
        if not flags.any():
            return 0
        lo = np.maximum(a + np.arange(n) * PAGE, off)
        hi = np.minimum(a + (np.arange(n) + 1) * PAGE, off + length)
        return int(((hi - lo) * flags).sum())

    def _scratch(self, n: int) -> mmap.mmap:
        buf = getattr(self._local, "buf", None)
        if buf is None or len(buf) < n:
            buf = mmap.mmap(-1, max(n, PIECE + 2 * ALIGN))          # anonymous: page-aligned
            self._local.buf = buf
        return buf

    def _piece(self, p0: int, p1: int, need: int, jobs: list[tuple[int, int, Any]]) -> dict[str, int | float]:
        res = self.residency(p0, min(p1, self.size) - p0) if self._base is not None else 0
        buf = self._scratch(p1 - p0)
        view = memoryview(buf)
        got = 0
        t0 = time.perf_counter()
        while got < p1 - p0:
            n = os.preadv(self.fd, [view[got:p1 - p0]], p0 + got)
            if n <= 0:
                break
            got += n
        dt = time.perf_counter() - t0
        if got < need - p0:
            raise OSError(f"{self.path}: short read at {p0}: {got} of {need - p0} bytes")
        src = np.frombuffer(buf, dtype=np.uint8, count=p1 - p0)
        for off, length, dst in jobs:
            lo, hi = max(off, p0), min(off + length, p1)
            if lo < hi:
                np.frombuffer(dst, dtype=np.uint8)[lo - off:hi - off] = src[lo - p0:hi - p0]
        del view, src
        return {"read_bytes": p1 - p0, "s": dt, "resident": res, "span": min(p1, self.size) - p0}

    def read(self, jobs: list[tuple[int, int, Any]]) -> dict[str, Any]:
        """Fill every job ``(file offset, bytes, writable buffer of that length)``; returns bytes, aligned bytes
        read, wall ms, and (with mincore) the reads and bytes that were cold or cached before the read."""

        t0 = time.perf_counter()
        jobs = sorted(jobs, key=lambda j: j[0])
        for off, length, dst in jobs:
            if off < 0 or off + length > self.size or memoryview(dst).nbytes != length:
                raise ValueError(f"{self.path}: bad read job ({off}, {length}) for a {self.size}-byte file")
        runs: list[list[Any]] = []
        for job in jobs:
            if runs and job[0] == runs[-1][1]:
                runs[-1][1] = job[0] + job[1]
                runs[-1][2].append(job)
            else:
                runs.append([job[0], job[0] + job[1], [job]])
        tasks = []
        for start, end, members in runs:
            a0, a1 = start // ALIGN * ALIGN, (end + ALIGN - 1) // ALIGN * ALIGN
            for p0 in range(a0, a1, PIECE):
                p1 = min(p0 + PIECE, a1)
                inside = [j for j in members if j[0] < p1 and j[0] + j[1] > p0]
                tasks.append((p0, p1, min(end, p1), inside))
        if self.pool is not None and len(tasks) > 1:
            done = list(self.pool.map(lambda t: self._piece(*t), tasks))
        else:
            done = [self._piece(*t) for t in tasks]
        out: dict[str, Any] = {"jobs": len(jobs), "reads": len(done), "bytes": sum(j[1] for j in jobs),
                               "read_bytes": sum(int(d["read_bytes"]) for d in done),
                               "read_ms": (time.perf_counter() - t0) * 1e3, "labeled": self._base is not None}
        if self._base is not None:
            cold = [d for d in done if int(d["resident"]) * 2 < int(d["span"])]
            out["cold_reads"], out["cached_reads"] = len(cold), len(done) - len(cold)
            out["cached_bytes"] = sum(int(d["resident"]) for d in done)
            out["cold_bytes"] = sum(int(d["span"]) for d in done) - out["cached_bytes"]
        return out

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
        if self._base is not None:
            _libc.munmap(ctypes.c_void_p(self._base), self.size)
            self._base = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _q2_jobs(plan: ProjPlan, experts: list[int]) -> tuple[np.ndarray, list[tuple[int, int, Any]]]:
    raw = np.empty((len(experts), plan.file_bytes), dtype=np.uint8)
    return raw, [(plan.offset + e * plan.file_bytes, plan.file_bytes, memoryview(raw[i])) for i, e in enumerate(experts)]


def _q2_parts(plan: ProjPlan, raw: np.ndarray) -> dict[str, np.ndarray]:
    from tensorfold.families.qwen4_exp.gguf_experts import q2_0_to_affine

    codes, scales, biases = q2_0_to_affine(raw.reshape(raw.shape[0], plan.rows, plan.k // 64 * 18), plan.k)
    return {"weight": codes, "scales": scales.view(np.uint16), "biases": biases.view(np.uint16)}


class _Buffer:
    """One ``kq.arena_alloc`` buffer of ``capacity`` items: the kernel side (``view``: zero-copy typed, shaped
    prefixes), its byte memoryview, and a host numpy view of the same bytes for writes."""

    def __init__(self, capacity: int, itemsize: int) -> None:
        import mlx_kquant as kq

        self.raw, self.mv = kq.arena_alloc([int(capacity)], itemsize)
        self.itemsize = itemsize
        self.host = np.frombuffer(self.mv, dtype={1: np.uint8, 2: np.uint16, 4: np.uint32}[itemsize])
        self.base = int(self.host.__array_interface__["data"][0])
        self.nbytes = int(capacity) * itemsize
        span = (self.nbytes + PAGE - 1) // PAGE * PAGE                  # arena_alloc maps whole pages
        if _libc.minherit(ctypes.c_void_p(self.base), span, VM_INHERIT_NONE) != 0:
            raise OSError(ctypes.get_errno(), "expert stream: minherit(VM_INHERIT_NONE) failed")

    def view(self, count: int, inner: tuple[int, ...], dtype: Any) -> mx.array:
        """The first ``count`` experts of shape ``inner`` as a [count, *inner] array of ``dtype`` (zero-copy)."""

        arr = self.raw[: count * int(np.prod(inner))].reshape(count, *inner)
        if dtype != arr.dtype:
            arr = arr.view(dtype)
        mx.eval(arr)
        if _address(arr) != self.base:
            raise RuntimeError("expert stream: an arena view is not zero-copy (MLX copied it)")
        return arr

    def slot(self, i: int, per: int) -> np.ndarray:
        return self.host[i * per:(i + 1) * per]

    def slot_bytes(self, i: int, per: int) -> memoryview:
        n = per * self.itemsize
        return self.mv[i * n:(i + 1) * n]


def _placeholder() -> mx.array:
    """kquant's vestigial ``scales``, evaluated here: MLX streams are per thread, and a lazy array made at load
    keeps the loading thread's stream, so its first evaluation on a server request thread raises "There is no
    Stream(gpu, 0) in current thread" (every array the stream builds must be materialized, as ``view`` does)."""

    arr = mx.zeros((1,), dtype=mx.uint8)
    mx.eval(arr)
    return arr


def _address(arr: mx.array) -> int:
    a = np.asarray(arr)
    if not a.flags["C_CONTIGUOUS"]:
        raise RuntimeError("expert stream: a module buffer is not contiguous")
    return int(a.__array_interface__["data"][0])


def pin_cap(memsize: int | None = None) -> float:
    """The most bytes ``Pin`` locks: PIN_FRACTION of ``memsize`` (default: this machine's physical memory), at most
    PIN_MAX_GIB GiB."""

    if memsize is None:
        try:
            memsize = os.sysconf("SC_PHYS_PAGES") * PAGE
        except (ValueError, OSError):
            memsize = 0
    cap = PIN_MAX_GIB * 2**30
    return float(min(cap, memsize * PIN_FRACTION)) if memsize > 0 else float(cap)


class Pin:
    """The buffers' host pages locked (``mlock``) and their Metal buffers in kquant's residency set; ``release`` undoes
    both. Over the cap it raises before locking anything; a failure part way releases what was pinned and raises
    (``ExpertStream`` then runs unpinned and says so)."""

    def __init__(self, bufs: list[_Buffer], max_bytes: float | None = None) -> None:
        import mlx_kquant as kq

        self.spans = [(b.base, (b.nbytes + PAGE - 1) // PAGE * PAGE) for b in bufs]      # arena_alloc: page-rounded
        self.nbytes = sum(n for _, n in self.spans)
        max_bytes = pin_cap() if max_bytes is None else max_bytes
        if self.nbytes > max_bytes:
            raise ValueError(f"expert stream: {self.nbytes / 2**30:.2f} GiB to pin (arena + ring) is over the "
                             f"{max_bytes / 2**30:.2f} GiB cap (TF_EXPERT_PIN=1)")
        self._bufs, self._locked, self._inserted = list(bufs), [], []
        try:
            for (base, n), buf in zip(self.spans, self._bufs):
                if _libc.mlock(ctypes.c_void_p(base), n) != 0:
                    raise OSError(ctypes.get_errno(), f"expert stream: mlock of {n} bytes failed "
                                                      f"({self.nbytes / 2**30:.2f} GiB to pin; vm.user_wire_limit?)")
                self._locked.append((base, n))
                if not kq.residency_insert(buf.raw):
                    raise RuntimeError("expert stream: kquant residency_insert refused an arena buffer")
                self._inserted.append(buf)
            if not kq.residency_commit():
                raise RuntimeError("expert stream: kquant residency_commit failed")
        except BaseException:
            self.release()
            raise

    @property
    def pinned(self) -> bool:
        return bool(self._locked)

    def release(self) -> None:
        import mlx_kquant as kq

        if self._inserted:
            for buf in self._inserted:
                kq.residency_erase(buf.raw)
            kq.residency_commit()
            self._inserted = []
        for base, n in self._locked:
            _libc.munlock(ctypes.c_void_p(base), n)
        self._locked = []


# -- the arena --------------------------------------------------------------------
def _per(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape))


def _check(switch: Any, expect: dict[str, dict[str, tuple[_Buffer, int]]]) -> int:
    """Each module buffer named in ``expect`` must start at its stream buffer and span the expected bytes."""

    n = 0
    for proj, parts in expect.items():
        mod = getattr(switch, proj)
        for name, (buf, nbytes) in parts.items():
            arr = mod[name]
            if _address(arr) != buf.base or arr.nbytes != nbytes:
                raise RuntimeError(f"expert stream: {proj}.{name} is not backed by the stream's buffer")
            n += 1
    return n


class LayerArena:
    """One layer's resident experts: C_L slots per projection in the stock layout, and the exclusive LRU over them."""

    def __init__(self, layer: int, plans: dict[str, ProjPlan], slots: int) -> None:
        self.layer, self.plans = layer, plans
        self.E = plans["gate_proj"].experts
        if any(p.experts != self.E for p in plans.values()):
            raise ValueError(f"layer {layer}: the projections hold different expert counts")
        self.C = min(int(slots), self.E)
        self.bufs: dict[str, dict[str, _Buffer]] = {}
        self._arrays: dict[str, dict[str, mx.array]] = {}
        for proj, plan in plans.items():
            self.bufs[proj], self._arrays[proj] = {}, {}
            for name, size, shape, dtype in plan.parts():
                buf = _Buffer(self.C * _per(shape), size)
                self.bufs[proj][name] = buf
                self._arrays[proj][name] = buf.view(self.C, shape, dtype)
            if plan.codec is not None:
                self._arrays[proj]["scales"] = _placeholder()
        self.expect = {proj: {name: (self.bufs[proj][name], self.C * _per(shape) * size)
                              for name, size, shape, _ in plan.parts()} for proj, plan in plans.items()}
        self.slot_of = np.full(self.E, -1, dtype=np.int64)
        self.expert_of = np.full(self.C, -1, dtype=np.int64)
        self.tick = np.full(self.C, -1, dtype=np.int64)
        self.clock = 0
        self.nbytes = sum(b.nbytes for parts in self.bufs.values() for b in parts.values())
        self.fixed = False               # a resident layer's arena: every expert in expert order, never written again

    def _writable(self) -> None:
        if self.fixed:
            raise RuntimeError(f"layer {self.layer}: a resident arena is never planned, evicted or written")

    def make_resident(self, reader: ExpertReader) -> dict[str, Any]:
        """Fill every slot once, expert e into slot e (C == E_L), then freeze the arena."""

        self._writable()
        if self.C != self.E:
            raise ValueError(f"layer {self.layer}: a resident arena needs {self.E} slots, it has {self.C}")
        self.slot_of[:] = self.expert_of[:] = np.arange(self.E)
        st = self.fill([(e, e) for e in range(self.E)], reader)
        self.fixed = True
        return st

    def arrays(self) -> dict[str, dict[str, mx.array]]:
        return self._arrays

    def resident(self) -> np.ndarray:
        return np.flatnonzero(self.slot_of >= 0)

    def _check_ids(self, ids: np.ndarray) -> None:
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.E):
            raise ValueError(f"layer {self.layer}: routed ids outside 0..{self.E - 1}")

    def _victim(self, protected: np.ndarray) -> int:
        s = int(np.argmin(np.where(protected, np.iinfo(np.int64).max, self.tick)))
        if protected[s]:
            raise ValueError(f"layer {self.layer}: no free slot (every slot holds an expert this call needs)")
        old = int(self.expert_of[s])
        if old >= 0:
            self.slot_of[old] = -1
        self.expert_of[s], self.tick[s] = -1, -1
        return s

    def plan(self, ids: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Slots for ``ids`` (router order) and the misses ``(slot, expert)`` to fill. A hit refreshes its slot; a
        miss takes the empty or least recently used slot whose expert this call does not route to."""

        self._writable()
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        self._check_ids(ids)
        uniq = np.unique(ids)
        if uniq.size > self.C:
            raise ValueError(f"layer {self.layer}: {uniq.size} distinct experts in one call, {self.C} slots")
        protected = np.zeros(self.C, dtype=bool)
        held = self.slot_of[uniq]
        protected[held[held >= 0]] = True
        slots = np.empty(ids.size, dtype=np.int64)
        misses: list[tuple[int, int]] = []
        for j, e in enumerate(ids.tolist()):
            self.clock += 1
            s = int(self.slot_of[e])
            if s < 0:
                s = self._victim(protected)
                self.slot_of[e], self.expert_of[s] = s, e
                protected[s] = True
                misses.append((s, e))
            self.tick[s] = self.clock
            slots[j] = s
        return slots, misses

    def touch(self, ids: np.ndarray) -> None:
        """Refresh the resident ``ids`` (router order) as ``plan`` would, without evicting anything."""

        self._writable()
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        self._check_ids(ids)
        if ids.size == 0:
            return
        uniq, first = np.unique(ids[::-1], return_index=True)
        last = ids.size - 1 - first                                     # each expert's last position
        s = self.slot_of[uniq]
        held = s >= 0
        self.tick[s[held]] = self.clock + 1 + last[held]
        self.clock += ids.size

    def invalidate(self, slots: list[int]) -> None:
        for s in slots:
            e = int(self.expert_of[s])
            if e >= 0:
                self.slot_of[e] = -1
            self.expert_of[s], self.tick[s] = -1, -1

    def fill(self, misses: list[tuple[int, int]], reader: ExpertReader) -> dict[str, Any]:
        """Read the misses' experts from the file into their slots (Q2_0 through the affine repack). On any
        failure the slots are marked empty before the error propagates."""

        self._writable()
        if not misses:
            return {}
        try:
            jobs: list[tuple[int, int, Any]] = []
            q2: list[tuple[str, ProjPlan, np.ndarray]] = []
            experts = [e for _, e in misses]
            for proj, plan in self.plans.items():
                if plan.codec is not None:
                    buf = self.bufs[proj]["weight"]
                    jobs += [(plan.offset + e * plan.file_bytes, plan.file_bytes, buf.slot_bytes(s, plan.file_bytes))
                             for s, e in misses]
                else:
                    raw, more = _q2_jobs(plan, experts)
                    jobs += more
                    q2.append((proj, plan, raw))
            st = reader.read(jobs)
            for proj, plan, raw in q2:
                parts = _q2_parts(plan, raw)                           # the whole repack before any slot write
                for name, _, shape, _ in plan.parts():
                    buf, arr = self.bufs[proj][name], parts[name]
                    for i, (s, _) in enumerate(misses):
                        buf.slot(s, _per(shape))[:] = arr[i].reshape(-1)
            return st
        except BaseException:
            self.invalidate([s for s, _ in misses])
            raise

    def admit(self, ordered: list[int], ring: Any) -> int:
        """Warm: make ``ordered`` (distinct, oldest first, at most C) resident in that recency order, copying the
        non-resident ones from the ring (which holds this layer's stack); returns the experts copied."""

        copies = self.assign(ordered)
        for proj, plan in self.plans.items():
            self.copy_in(proj, copies, {name: ring.buf(plan, name) for name, *_ in plan.parts()})
        return len(copies)

    def assign(self, ordered: list[int]) -> list[tuple[int, int]]:
        """``admit``'s slot bookkeeping alone: the ``(slot, expert)`` copies it needs, the slots already assigned (their
        bytes still to be copied, projection by projection, with ``copy_in``)."""

        self._writable()
        protected = np.zeros(self.C, dtype=bool)
        held = self.slot_of[np.asarray(ordered, dtype=np.int64)]
        protected[held[held >= 0]] = True
        copies: list[tuple[int, int]] = []
        for e in ordered:
            self.clock += 1
            s = int(self.slot_of[e])
            if s < 0:
                s = self._victim(protected)
                self.slot_of[e], self.expert_of[s] = s, e
                protected[s] = True
                copies.append((s, e))
            self.tick[s] = self.clock
        return copies

    def copy_in(self, proj: str, copies: list[tuple[int, int]], src: dict[str, "_Buffer"],
                position: np.ndarray | None = None) -> None:
        """One projection's bytes of ``copies`` from the buffers ``src`` (by part; expert e at its own position, or
        at ``position[e]`` in a compact stack) into the arena."""

        self._writable()
        plan = self.plans[proj]
        for name, _, shape, _ in plan.parts():
            per = _per(shape)
            for s, e in copies:
                at = e if position is None else int(position[e])
                self.bufs[proj][name].slot(s, per)[:] = src[name].slot(at, per)


# -- the prefill ring -------------------------------------------------------------
def _stage_jobs(arena: LayerArena, proj: str, dst: dict[str, _Buffer], hits: list[tuple[int, int]],
                misses: list[tuple[int, int]]) -> tuple[list[tuple[int, int, Any]], Any]:
    """One projection into the buffers ``dst``: hits ``(position, arena slot)`` copied now; returns the read jobs of the
    misses ``(position, expert)`` and the Q2_0 repack to finish after the read (``_finish_q2``), or None."""

    plan = arena.plans[proj]
    for name, _, shape, _ in plan.parts():
        per = _per(shape)
        for pos, s in hits:
            dst[name].slot(pos, per)[:] = arena.bufs[proj][name].slot(s, per)
    if not misses:
        return [], None
    if plan.codec is not None:
        buf = dst["weight"]
        return [(plan.offset + e * plan.file_bytes, plan.file_bytes, buf.slot_bytes(pos, plan.file_bytes))
                for pos, e in misses], None
    raw, jobs = _q2_jobs(plan, [e for _, e in misses])
    return jobs, (plan, dst, raw, [pos for pos, _ in misses])


def _finish_q2(post: Any) -> None:
    plan, dst, raw, positions = post
    parts = _q2_parts(plan, raw)                                   # the whole repack before any write
    for name, _, shape, _ in plan.parts():
        per = _per(shape)
        for i, pos in enumerate(positions):
            dst[name].slot(pos, per)[:] = parts[name][i].reshape(-1)


def _add_reads(st: dict[str, Any], more: dict[str, Any]) -> None:
    for k, v in more.items():
        if k != "labeled":
            st[k] = st.get(k, 0) + v


class PrefillRing:
    """Full-layout stacks for the calls the arena does not take: one buffer per (projection, codec family, part),
    sized for the largest layer that uses it and viewed per layer as that layer's [E_L, ...] stack."""

    shared = False

    def __init__(self, arenas: list[LayerArena]) -> None:
        need: dict[tuple, tuple[int, int]] = {}
        for arena in arenas:
            for plan in arena.plans.values():
                for name, size, shape, _ in plan.parts():
                    key = self.kind(plan, name)
                    items = plan.experts * _per(shape)
                    need[key] = (max(items, need.get(key, (0, 0))[0]), size)
        self.bufs = {key: _Buffer(items, size) for key, (items, size) in need.items()}
        self.nbytes = sum(b.nbytes for b in self.bufs.values())       # what is allocated, every kind summed
        self._views: dict[int, dict[str, dict[str, mx.array]]] = {}
        self._placeholder = _placeholder()

    @staticmethod
    def kind(plan: ProjPlan, name: str) -> tuple:
        return (plan.proj, plan.codec is None, name)

    def buf(self, plan: ProjPlan, name: str) -> _Buffer:
        return self.bufs[self.kind(plan, name)]

    def expect(self, arena: LayerArena) -> dict[str, dict[str, tuple[_Buffer, int]]]:
        return {proj: {name: (self.buf(plan, name), plan.experts * _per(shape) * size)
                       for name, size, shape, _ in plan.parts()} for proj, plan in arena.plans.items()}

    def arrays(self, arena: LayerArena) -> dict[str, dict[str, mx.array]]:
        views = self._views.get(arena.layer)
        if views is None:
            views = {}
            for proj, plan in arena.plans.items():
                views[proj] = {name: self.buf(plan, name).view(plan.experts, shape, dtype)
                               for name, _, shape, dtype in plan.parts()}
                if plan.codec is not None:
                    views[proj]["scales"] = self._placeholder
            self._views[arena.layer] = views
        return views

    def stage(self, arena: LayerArena, uniq: np.ndarray, reader: ExpertReader) -> dict[str, Any]:
        """Put every expert of ``uniq`` (distinct storage ids) at its own position: resident ones copied from the
        arena, the rest read from the file."""

        held = arena.slot_of[uniq]
        hits = list(zip(uniq[held >= 0].tolist(), held[held >= 0].tolist()))
        miss = np.sort(uniq[held < 0]).tolist()
        st: dict[str, Any] = {"copied": len(hits), "misses": len(miss)}
        jobs: list[tuple[int, int, Any]] = []
        q2 = []
        for proj, plan in arena.plans.items():
            more, post = _stage_jobs(arena, proj, {name: self.buf(plan, name) for name, *_ in plan.parts()}, hits,
                                     [(e, e) for e in miss])
            jobs += more
            if post is not None:
                q2.append(post)
        if jobs:
            st.update(reader.read(jobs))
            for post in q2:
                _finish_q2(post)
        return st


class SharedRing:
    """TF_EXPERT_RING=shared: one buffer per (codec family, part), sized for the largest single projection stack,
    viewed per (layer, projection) as that projection's [E_L, ...] stack (the split ring's views, one projection at a
    time); plus the pair buffer for the up stack of fused-GLU unsorted calls (``compact``)."""

    shared = True

    def __init__(self, arenas: list[LayerArena]) -> None:
        import mlx_kquant as kq

        need: dict[tuple, tuple[int, int]] = {}
        pair = 0
        self.fusable: dict[int, int] = {}            # layer -> the compact stacks' expert count (fused GLU layers)
        for arena in arenas:
            for plan in arena.plans.values():
                for name, size, shape, _ in plan.parts():
                    key = self.kind(plan, name)
                    items = plan.experts * _per(shape)
                    need[key] = (max(items, need.get(key, (0, 0))[0]), size)
            gate, up = arena.plans["gate_proj"], arena.plans["up_proj"]
            if gate.codec is not None and gate.codec == up.codec and kq.codec_has_moe_glu(gate.codec):
                self.fusable[arena.layer] = min(SORT_MIN - 1, arena.E)
                pair = max(pair, self.fusable[arena.layer] * _per(up.parts()[0][2]))
        self.bufs = {key: _Buffer(items, size) for key, (items, size) in need.items()}
        if pair:
            self.bufs[("pair", "weight")] = _Buffer(pair, 1)
        self.nbytes = sum(b.nbytes for b in self.bufs.values())
        self._views: dict[int, dict[str, dict[str, mx.array]]] = {}
        self._compact: dict[int, dict[str, dict[str, mx.array]]] = {}
        self._placeholder = _placeholder()

    @staticmethod
    def kind(plan: ProjPlan, name: str) -> tuple:
        return (plan.codec is None, name)

    def buf(self, plan: ProjPlan, name: str) -> _Buffer:
        return self.bufs[self.kind(plan, name)]

    def dst(self, plan: ProjPlan, compact: bool) -> dict[str, _Buffer]:
        """The buffers a stage of ``plan`` writes: the shared ones, or for a compact up stack the pair buffer."""

        if compact and plan.proj == "up_proj":
            return {"weight": self.bufs[("pair", "weight")]}
        return {name: self.buf(plan, name) for name, *_ in plan.parts()}

    def expect(self, arena: LayerArena) -> dict[str, dict[str, tuple[_Buffer, int]]]:
        return {proj: {name: (self.buf(plan, name), plan.experts * _per(shape) * size)
                       for name, size, shape, _ in plan.parts()} for proj, plan in arena.plans.items()}

    def _make(self, arena: LayerArena, count: int, compact: bool) -> dict[str, dict[str, mx.array]]:
        views: dict[str, dict[str, mx.array]] = {}
        for proj, plan in arena.plans.items():
            dst = self.dst(plan, compact)
            views[proj] = {name: dst[name].view(count, shape, dtype) for name, _, shape, dtype in plan.parts()}
            if plan.codec is not None:
                views[proj]["scales"] = self._placeholder
        return views

    def arrays(self, arena: LayerArena) -> dict[str, dict[str, mx.array]]:
        views = self._views.get(arena.layer)
        if views is None:
            views = self._views[arena.layer] = self._make(arena, arena.E, False)
        return views

    def compact(self, arena: LayerArena) -> dict[str, dict[str, mx.array]]:
        """The fused-GLU layer's compact stacks (``fusable[layer]`` experts each; gate and down shared, up the pair)."""

        views = self._compact.get(arena.layer)
        if views is None:
            views = self._compact[arena.layer] = self._make(arena, self.fusable[arena.layer], True)
        return views


# -- the stream -------------------------------------------------------------------
_TOTALS = ("misses", "bytes", "read_bytes", "reads", "read_ms", "wait_ms", "sync_ms", "settle_ms", "cold_reads",
           "cached_reads", "cold_bytes", "cached_bytes", "copied", "warmed")


class ExpertStream:
    """Every streamed layer's arena, the ring, the reader and the counters; ``layer(i)`` is what a SparseMoE calls."""

    def __init__(self, experts: Any, cfg: dict[str, Any], layers: list[int] | None = None) -> None:
        plans = plans_from(experts)
        if layers is not None:
            plans = [plans[i] for i in layers]
        self.cfg = dict(cfg)
        known = [p["gate_proj"].layer for p in plans]
        self.resident = [int(i) for i in cfg.get("resident") or []]
        self.layer_slots = {int(k): int(v) for k, v in (cfg.get("layer_slots") or {}).items()}
        for name, named in (("TF_EXPERT_RESIDENT", self.resident), ("TF_EXPERT_LAYER_SLOTS", list(self.layer_slots))):
            unknown = sorted(set(named) - set(known))
            if unknown:
                raise ValueError(f"{name}: layers {unknown} are not streamed layers {known}")
        if set(self.resident) & set(self.layer_slots):
            raise ValueError("TF_EXPERT_RESIDENT and TF_EXPERT_LAYER_SLOTS name the same layer")
        if any(c < MIN_SLOTS for c in self.layer_slots.values()):
            raise ValueError(f"TF_EXPERT_LAYER_SLOTS: at least {MIN_SLOTS} slots a layer")
        own = {p["gate_proj"].layer: p["gate_proj"].experts for p in plans if p["gate_proj"].layer in self.resident}
        own.update(self.layer_slots)
        equal = [p for p in plans if p["gate_proj"].layer not in own]
        slots = cfg.get("slots")
        if slots is None:
            fixed = sum(arena_bytes([p], own[p["gate_proj"].layer]) for p in plans if p["gate_proj"].layer in own)
            slots = slots_for_gib(equal, float(cfg["gib"]), fixed) if equal else 0
        self.slots = int(slots)
        if equal and self.slots < MIN_SLOTS:
            raise ValueError(f"expert stream: {self.slots} slots a layer, at least {MIN_SLOTS}")
        self.arenas = [LayerArena(p["gate_proj"].layer, p, own.get(p["gate_proj"].layer, self.slots)) for p in plans]
        self.index = {a.layer: i for i, a in enumerate(self.arenas)}      # every layer, resident ones too (the order)
        streamed = [a for a in self.arenas if a.layer not in self.resident]
        self.ring = SharedRing(streamed) if cfg.get("ring") == "shared" else PrefillRing(streamed)
        self.pin: Pin | None = None
        self.pin_refused: str | None = None
        if cfg.get("pin"):
            try:
                self.pin = Pin([b for a in self.arenas for parts in a.bufs.values() for b in parts.values()]
                               + list(self.ring.bufs.values()))
            except (ValueError, OSError, RuntimeError) as exc:
                # over the cap, or mlock or the residency set refused: Pin released what it had taken
                self.pin_refused = str(exc)
                print(f"tensorfold: expert stream: not pinned, running unpinned ({exc})", flush=True)
        self.reader = ExpertReader(experts.path, int(cfg.get("threads", 8)), bool(cfg.get("mincore")))
        self.resident_calls = 0
        self.resident_load: dict[str, Any] = {}
        try:
            for a in self.arenas:
                if a.layer in self.resident:
                    _add_reads(self.resident_load, a.make_resident(self.reader))
        except BaseException:
            self.reader.close()
            if self.pin is not None:
                self.pin.release()
            raise
        self.warm_rows = int(cfg.get("warm_rows", 64))
        self.prefill_rows = int(cfg.get("prefill_rows", 2048))
        self.debug = bool(cfg.get("debug"))
        self.fence = True                 # tests turn it off to show the hazard it closes
        self.final_chunk: bool | None = None
        self.pending: mx.array | None = None
        self.calls: deque[dict[str, Any]] = deque(maxlen=CALLS_KEPT)
        self.totals: dict[str, Any] = {}
        self._stats = open(cfg["stats"], "a") if cfg.get("stats") else None
        self._next: int | None = None
        self._owner: int | None = None

    @property
    def arena_bytes(self) -> int:
        return sum(a.nbytes for a in self.arenas)

    @property
    def resident_bytes(self) -> int:
        return sum(a.nbytes for a in self.arenas if a.fixed)

    def layer(self, i: int) -> "LayerStream | ResidentLayer":
        arena = self.arenas[self.index[i]]
        return ResidentLayer(self, arena) if arena.fixed else LayerStream(self, arena)

    def summary(self) -> str:
        extra = ""
        if self.resident:
            extra += f", resident layers {self.resident} ({self.resident_bytes / 2**30:.2f} GiB)"
        if self.layer_slots:
            extra += ", layer slots " + " ".join(f"{k}:{v}" for k, v in sorted(self.layer_slots.items()))
        return (f"tensorfold: expert stream: {self.slots} slots a layer ({self.arena_bytes / 2**30:.2f} GiB arena)"
                f"{extra}, ring {self.ring.nbytes / 2**20:.0f} MiB {'shared' if self.ring.shared else 'split'}, io threads {self.reader.threads}, "
                f"mincore {'on' if self.reader.mincore else 'off'}, "
                f"{f'pinned {self.pin.nbytes / 2**30:.2f} GiB' if self.pin is not None else 'unpinned'}"
                f"{' (pin refused)' if self.pin_refused else ''}")

    def module_arrays(self) -> dict[str, mx.array]:
        """The arena arrays by checkpoint path (``model.layers.L.mlp.switch_mlp.P.{weight,scales[,biases]}``)."""

        return {f"model.layers.{a.layer}.mlp.switch_mlp.{proj}.{name}": arr
                for a in self.arenas for proj, parts in a.arrays().items() for name, arr in parts.items()}

    def check_views(self, model: Any) -> int:
        """Every streamed module's buffers must be the arena's (same address, same bytes, contiguous): an MLX copy
        would leave the kernels reading memory the stream never writes. Returns the buffers checked."""

        return sum(_check(model.layers[a.layer].mlp.switch_mlp, a.expect) for a in self.arenas)

    # -- per call
    def _enter(self, layer: int) -> None:
        me, pos = threading.get_ident(), self.index[layer]
        if pos == 0:
            self._owner = me
            self.flush()
        elif self._next != pos or self._owner != me:
            self._next = None
            raise RuntimeError(f"expert stream: layer {layer} called out of order or from another thread; one "
                               "forward at a time is supported (no concurrent forwards, no batched lanes)")
        self._next = (pos + 1) % len(self.arenas)

    def done(self, out: mx.array) -> None:
        """The layer's MoE output: the next layer call evaluates it before any slot or ring write."""

        self.pending = out

    def settle(self) -> None:
        if self.fence and self.pending is not None:
            mx.eval(self.pending)
        self.pending = None

    def is_final(self, rows: int) -> bool:
        return self.final_chunk if self.final_chunk is not None else rows < self.prefill_rows

    def record(self, rec: dict[str, Any]) -> None:
        self.calls.append(rec)
        for k in _TOTALS:
            if k in rec:
                self.totals[k] = self.totals.get(k, 0) + rec[k]
        key = f"calls_{rec['path']}"
        self.totals[key] = self.totals.get(key, 0) + 1

    def drain(self) -> list[dict[str, Any]]:
        """The calls recorded since the last drain (also appended to TF_EXPERT_STATS when set)."""

        calls = list(self.calls)
        self.calls.clear()
        if self._stats is not None and calls:
            self._stats.write("".join(json.dumps(c) + "\n" for c in calls))
            self._stats.flush()
        return calls

    def flush(self) -> None:
        if self._stats is not None:
            self.drain()

    def close(self) -> None:
        self.flush()
        if self._stats is not None:
            self._stats.close()
            self._stats = None
        self.reader.close()
        if self.pin is not None:
            self.pin.release()


class LayerStream:
    """``SparseMoE.__call__``'s hook for one layer: sync on the routed ids, take the arena or the ring path, run the
    stock switch over it and return its output (the caller hands the MoE output back through ``done``)."""

    def __init__(self, stream: ExpertStream, arena: LayerArena) -> None:
        self.stream, self.arena = stream, arena

    @staticmethod
    def _install(switch: Any, arrays: dict[str, dict[str, mx.array]]) -> None:
        for proj, parts in arrays.items():
            mod = getattr(switch, proj)
            for name, arr in parts.items():
                if mod[name] is not arr:
                    setattr(mod, name, arr)

    def done(self, out: mx.array) -> None:
        self.stream.done(out)

    def _warm_order(self, ids: np.ndarray, experts: mx.array) -> list[int]:
        """The experts the call's last TF_EXPERT_WARM_ROWS rows routed: distinct, oldest first, the newest C."""

        rev = ids[-self.stream.warm_rows * experts.shape[-1]:][::-1]
        _, first = np.unique(rev, return_index=True)
        return rev[np.sort(first)][::-1][-self.arena.C:].tolist()

    def _shared(self, switch: Any, x: mx.array, experts: mx.array, ids: np.ndarray, uniq: np.ndarray, warm: bool,
                rec: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """The ring call over the shared ring: ``SwitchGLU.__call__``'s ops (or ``GGUFSwitchGLU``'s fused branch) in
        their order, each projection staged just before its kernel, after the eval of every earlier reader."""

        s, arena, ring = self.stream, self.arena, self.stream.ring
        held = arena.slot_of[uniq]                                      # hits and misses, fixed before any warming
        hit_e, hit_s = uniq[held >= 0].tolist(), held[held >= 0].tolist()
        miss = np.sort(uniq[held < 0]).tolist()
        copies = arena.assign(self._warm_order(ids, experts)) if warm else None
        st: dict[str, Any] = {"copied": len(hit_e), "misses": len(miss)}
        codec = switch.__dict__.get("fused_codec")
        compact = (codec is not None and ids.size < SORT_MIN and os.environ.get("TF_FLASH_KQ_GLU", "1") != "0")
        position = None
        if compact:
            position = np.full(arena.E, -1, dtype=np.int64)
            position[uniq] = np.arange(uniq.size)
            views = ring.compact(arena)
        else:
            views = ring.arrays(arena)

        def stage(proj: str) -> None:
            plan = arena.plans[proj]
            at = (lambda e: e) if position is None else (lambda e: int(position[e]))
            dst = ring.dst(plan, compact)
            jobs, post = _stage_jobs(arena, proj, dst, [(at(e), sl) for e, sl in zip(hit_e, hit_s)],
                                     [(at(e), e) for e in miss])
            if jobs:
                _add_reads(st, s.reader.read(jobs))
                if post is not None:
                    _finish_q2(post)
            if copies is not None:
                arena.copy_in(proj, copies, dst, position)              # warm while this projection is staged
            self._install(switch, {proj: views[proj]})
            if s.debug and not compact:
                _check(switch, {proj: ring.expect(arena)[proj]})

        def fence(a: mx.array) -> None:
            if s.fence:
                mx.eval(a)                                              # the next stage overwrites what ``a`` reads

        if compact:
            # GGUFSwitchGLU's fused branch over compact stacks (gate and up at once), ids remapped as the arena's
            import mlx_kquant as kq

            stage("gate_proj")
            stage("up_proj")
            lead, top = experts.shape[:-1], experts.shape[-1]
            idx = mx.array(position[ids].reshape(-1, top).astype(np.uint32))
            h = kq.moe_glu_gather_kq(x.reshape(-1, x.shape[-1]), switch.gate_proj["weight"], switch.up_proj["weight"],
                                     codec, idx)                                   # [T, R, N]
            fence(h)
            stage("down_proj")
            y = switch.down_proj(h[:, :, None, :], idx, sorted_indices=False)      # [T, R, 1, D]
            y = y.reshape(*lead, top, y.shape[-1])
        else:
            # mlx-lm SwitchGLU.__call__, one projection staged at a time
            xs = mx.expand_dims(x, (-2, -3))
            do_sort = experts.size >= SORT_MIN
            idx, inv_order = experts, None
            if do_sort:
                xs, idx, inv_order = _gather_sort(xs, experts)
            stage("up_proj")
            x_up = switch.up_proj(xs, idx, sorted_indices=do_sort)
            fence(x_up)
            stage("gate_proj")
            x_gate = switch.gate_proj(xs, idx, sorted_indices=do_sort)
            fence(x_gate)
            stage("down_proj")
            y = switch.down_proj(switch.activation(x_up, x_gate), idx, sorted_indices=do_sort)
            if do_sort:
                y = _scatter_unsort(y, inv_order, experts.shape)
            y = y.squeeze(-2)
        rec.update(compact=compact)
        if copies is not None:
            rec["warmed"] = len(copies)
        return y, st

    def __call__(self, moe: Any, x: mx.array, experts: mx.array) -> mx.array:
        s, arena = self.stream, self.arena
        if x.ndim >= 3 and x.shape[0] != 1:
            raise RuntimeError(f"expert stream: batch {x.shape[0]}; one sequence at a time (no batched lanes)")
        s._enter(arena.layer)
        t0 = time.perf_counter()
        ids = np.array(experts).reshape(-1).astype(np.int64)            # the host sync (up to this layer's router)
        t1 = time.perf_counter()
        s.settle()                                                      # every earlier reader of arena and ring
        t2 = time.perf_counter()
        rows = int(np.prod(experts.shape[:-1]))
        rec: dict[str, Any] = {"layer": arena.layer, "rows": rows, "ids": int(ids.size)}
        uniq = np.unique(ids)
        switch = moe.switch_mlp
        if ids.size < SORT_MIN and uniq.size <= arena.C:
            slots, misses = arena.plan(ids)
            st = arena.fill(misses, s.reader)
            self._install(switch, arena.arrays())
            if s.debug:
                _check(switch, arena.expect)
            y = switch(x, mx.array(slots.reshape(experts.shape).astype(np.uint32)))
            rec.update(path="arena", misses=len(misses))
        else:
            arena.touch(ids)
            warm = s.warm_rows > 0 and s.is_final(rows)
            rec.update(path="ring", uniq=int(uniq.size))
            if s.ring.shared:
                y, st = self._shared(switch, x, experts, ids, uniq, warm, rec)
            else:
                st = s.ring.stage(arena, uniq, s.reader)
                self._install(switch, s.ring.arrays(arena))
                if s.debug:
                    _check(switch, s.ring.expect(arena))
                y = switch(x, experts)
                if warm:
                    rec["warmed"] = arena.admit(self._warm_order(ids, experts), s.ring)
        t3 = time.perf_counter()
        rec.update({k: v for k, v in (st or {}).items() if k != "labeled"})
        rec.update(sync_ms=(t1 - t0) * 1e3, settle_ms=(t2 - t1) * 1e3, wait_ms=(t3 - t0) * 1e3)
        s.record(rec)
        return y


class ResidentLayer:
    """A resident layer's hook (TF_EXPERT_RESIDENT): the stock switch call over the frozen arena (slot i is expert i),
    with the router's ids untouched. Nothing here waits on the GPU: no host sync, no plan, no ring, no fence."""

    def __init__(self, stream: ExpertStream, arena: LayerArena) -> None:
        if not arena.fixed:
            raise RuntimeError(f"expert stream: layer {arena.layer} is not resident")
        self.stream, self.arena = stream, arena

    def done(self, out: mx.array) -> None:
        """No-op: the fence stays the last streamed layer's output (this layer's buffers are never written)."""

    def __call__(self, moe: Any, x: mx.array, experts: mx.array) -> mx.array:
        s, switch = self.stream, moe.switch_mlp
        if x.ndim >= 3 and x.shape[0] != 1:
            raise RuntimeError(f"expert stream: batch {x.shape[0]}; one sequence at a time (no batched lanes)")
        s._enter(self.arena.layer)
        LayerStream._install(switch, self.arena.arrays())           # already the modules' arrays after load
        if s.debug:
            _check(switch, self.arena.expect)
        s.resident_calls += 1
        return switch(x, experts)
