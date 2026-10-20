"""Streamed experts (expert_stream.py, TF_EXPERT_STREAM=1) against the all-resident build, on a synthetic GGUF.

The file holds four layers at the real expert dims (K 2560, N 640, 24 experts a layer), one per codec pairing the
cut240 file uses: IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS gate/up over IQ4_NL or Q2_0 down. Block bytes are random with a
finite fp16 scale in each block's first two bytes (every one of these block formats starts with it), so experts are
distinct and outputs finite. Arenas hold 16 slots (the production floor) and calls route 10 distinct experts, as
the model does. Outputs are compared bit for bit (bf16 bits), never with a tolerance. GPU kernels, small arrays.
"""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
kq = pytest.importorskip("mlx_kquant")

from tensorfold.families.qwen4_exp import expert_stream as es  # noqa: E402
from tensorfold.families.qwen4_exp.gguf_experts import (  # noqa: E402
    _TYPES, AffineQ2SwitchLinear, GGUFExperts, _switch_glu_class, q2_0_to_affine, read_gguf_tensors)

K, N, E, TOP, SLOTS = 2560, 640, 24, 10, 16
TYPE_ID = {v[0]: k for k, v in _TYPES.items()}
# (gate/up, down) per layer: every expert codec of the cut240 file, and both down kinds
LAYERS = [("IQ2_S", "IQ4_NL"), ("IQ3_XXS", "Q2_0"), ("IQ3_S", "IQ4_NL"), ("IQ4_XS", "Q2_0")]
PROJ_SHORT = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


# -- a synthetic GGUF ---------------------------------------------------------------
def _blocks(rng: np.random.Generator, type_name: str, k: int, rows: int) -> np.ndarray:
    _, _, wpb, bpb = _TYPES[TYPE_ID[type_name]]
    raw = rng.integers(0, 256, size=(rows * (k // wpb), bpb), dtype=np.uint8)
    d = rng.uniform(0.0002, 0.002, size=raw.shape[0]).astype(np.float16)
    raw[:, :2] = d.view(np.uint8).reshape(-1, 2)
    return raw.reshape(-1)


def write_gguf(path: Path, tensors: list[tuple[str, tuple[int, ...], int, bytes]], align: int = 32) -> None:
    """GGUF v3: one key (general.alignment), tensor infos, aligned data."""

    def s(text: str) -> bytes:
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1) + s("general.alignment") + struct.pack("<II", 4, align)
    rel, offsets = 0, []
    for _, _, _, data in tensors:
        offsets.append(rel)
        rel = (rel + len(data) + align - 1) // align * align
    for (name, dims, ggml_type, _), off in zip(tensors, offsets):
        head += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", ggml_type, off)
    with open(path, "wb") as f:
        f.write(head + b"\0" * ((-len(head)) % align))
        for (_, _, _, data), off in zip(tensors, offsets):
            f.write(data)
            f.write(b"\0" * ((-len(data)) % align))


def expert_tensors(rng: np.random.Generator, layers: list[tuple[str, str]], k: int, n: int, e: int):
    out = []
    for li, (gu, down) in enumerate(layers):
        for short, kind, kk, nn_ in (("gate", gu, k, n), ("up", gu, k, n), ("down", down, n, k)):
            out.append((f"blk.{li}.ffn_{short}_exps.weight", (kk, nn_, e), TYPE_ID[kind],
                        _blocks(rng, kind, kk, nn_ * e).tobytes()))
    return out


@pytest.fixture(scope="module")
def gguf(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("stream") / "experts.gguf"
    write_gguf(path, expert_tensors(np.random.default_rng(7), LAYERS, K, N, E))
    return path


def experts_of(path: Path, layers: int = len(LAYERS), k: int = K, n: int = N, e: int = E) -> GGUFExperts:
    return GGUFExperts(path.parent, {"format": "gguf-kquant", "file": str(path)}, layers, e, k, n)


@pytest.fixture(scope="module")
def stock(gguf):
    """The all-resident arrays (kq.load_gguf zero-copy + the Q2_0 repack), as the stock loader builds them."""

    return experts_of(gguf).arrays()


def make_switch(gx: GGUFExperts, layer: int, arrays: dict[str, mx.array] | None, experts: int):
    from mlx_kquant.nn import KQuantSwitchLinear

    made = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        out_dims, in_dims = (gx.intermediate, gx.hidden) if proj != "down_proj" else (gx.hidden, gx.intermediate)
        codec = gx.codec(layer, proj)
        mod = (KQuantSwitchLinear(experts, out_dims, in_dims, False, codec) if codec
               else AffineQ2SwitchLinear(experts, out_dims, in_dims))
        if arrays is not None:
            for key, value in arrays.items():
                prefix = f"model.layers.{layer}.mlp.switch_mlp.{proj}."
                if key.startswith(prefix):
                    setattr(mod, key[len(prefix):], value)
        made[proj] = mod
    return _switch_glu_class()(made["gate_proj"], made["up_proj"], made["down_proj"])


class FakeMoE:
    def __init__(self, switch):
        self.switch_mlp = switch


def bits(a: mx.array) -> np.ndarray:
    mx.eval(a)
    return np.array(a.view(mx.uint16) if a.dtype in (mx.bfloat16, mx.float16) else a)


def cfg(**kw):
    out = {"slots": SLOTS, "gib": None, "threads": 4, "warm_rows": 64, "prefill_rows": 2048, "mincore": False,
           "stats": None, "debug": True}
    out.update(kw)
    return out


def route(rng, rows: int, pool: np.ndarray | None = None) -> np.ndarray:
    pool = np.arange(E) if pool is None else pool
    return np.stack([rng.choice(pool, TOP, replace=False) for _ in range(rows)]).astype(np.uint32)[None]


def x_of(rng, rows: int) -> mx.array:
    return mx.array(rng.standard_normal((1, rows, K)).astype(np.float32)).astype(mx.bfloat16)


@pytest.fixture(autouse=True)
def _small():
    yield
    assert mx.get_active_memory() < 2**30, "the tests must stay well under 1 GiB of GPU memory"


# -- 0. the synthetic file ------------------------------------------------------------
def test_synthetic_gguf_reads_with_both_readers(gguf, stock):
    tensors = read_gguf_tensors(gguf)
    assert len(tensors) == 3 * len(LAYERS)
    t = tensors["blk.1.ffn_down_exps.weight"]
    assert t.dims == (N, K, E) and _TYPES[t.ggml_type][0] == "Q2_0"
    t = tensors["blk.0.ffn_gate_exps.weight"]
    per = K // 256 * 82 * N
    with open(gguf, "rb") as f:
        f.seek(t.offset + 5 * per)
        raw = np.frombuffer(f.read(per), dtype=np.uint8)
    w = stock["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    assert w.shape == (E, N, K // 256 * 82)
    assert np.array_equal(np.array(w[5]).reshape(-1), raw)


# -- 1-2. bookkeeping -----------------------------------------------------------------
class RefLRU:
    """The exclusive LRU in plain Python: slots by index, empty slots first (lowest index), victims the least
    recently used slot whose expert the call does not route to."""

    def __init__(self, slots: int):
        self.expert = [-1] * slots
        self.order: OrderedDict[int, None] = OrderedDict()      # slots, least recent first

    def call(self, ids: list[int]):
        where = {e: s for s, e in enumerate(self.expert) if e >= 0}
        need = set(ids)
        out, misses = [], []
        for e in ids:
            if e in where:
                s = where[e]
            else:
                empty = [s for s, x in enumerate(self.expert) if x < 0]
                if empty:
                    s = empty[0]
                else:
                    s = next(s for s in self.order if self.expert[s] not in need)
                if self.expert[s] >= 0:
                    del where[self.expert[s]]
                self.expert[s] = e
                where[e] = s
                misses.append((s, e))
            self.order.pop(s, None)
            self.order[s] = None
            out.append(s)
        return out, misses


def plain_lru(seq, cap: int) -> np.ndarray:
    """e1a_replay.lru: ids one at a time, any resident expert may be evicted."""

    cache: OrderedDict[int, None] = OrderedDict()
    miss = np.zeros(len(seq), dtype=np.int64)
    for t, row in enumerate(seq):
        for e in row:
            e = int(e)
            if e in cache:
                cache.move_to_end(e)
            else:
                miss[t] += 1
                cache[e] = None
                if len(cache) > cap:
                    cache.popitem(last=False)
    return miss


def tiny_arena(experts: int, slots: int) -> es.LayerArena:
    plans = {p: es.ProjPlan(0, p, "iq4_nl", 0, experts, 1, 32, 18) for p in ("gate_proj", "up_proj", "down_proj")}
    return es.LayerArena(0, plans, slots)


def test_arena_lru_matches_the_reference_token_by_token():
    rng = np.random.default_rng(3)
    arena, ref = tiny_arena(40, 16), RefLRU(16)
    for t in range(400):
        rows = 1 if t % 5 else int(rng.integers(2, 4))
        pool = rng.choice(40, 16, replace=False)
        ids = np.concatenate([rng.choice(pool, 5, replace=False) for _ in range(rows)])
        slots, misses = arena.plan(ids)
        want_slots, want_misses = ref.call(ids.tolist())
        assert slots.tolist() == want_slots and misses == want_misses, t
        assert all(arena.expert_of[s] == e for s, e in zip(slots, ids))
    with pytest.raises(ValueError):
        arena.plan(np.array([40]))
    with pytest.raises(ValueError):
        arena.plan(np.arange(17))


def test_exclusive_lru_equals_plain_lru_unless_the_rule_bites():
    rng = np.random.default_rng(4)
    seq = [rng.choice(64, TOP, replace=False) for _ in range(600)]
    for cap, may_differ in ((60, True), (16, True)):
        arena = tiny_arena(64, cap)
        mine = np.array([len(arena.plan(np.array(r))[1]) for r in seq])
        plain = plain_lru(seq, cap)
        # the rule bites when plain LRU evicts an expert the same row still needs: replay that and compare
        bites = np.zeros(len(seq), dtype=bool)
        cache: OrderedDict[int, None] = OrderedDict()
        for t, row in enumerate(seq):
            need = set(int(e) for e in row)
            for e in row:
                e = int(e)
                if e in cache:
                    cache.move_to_end(e)
                    continue
                cache[e] = None
                if len(cache) > cap:
                    victim, _ = cache.popitem(last=False)
                    bites[t] |= victim in need
        first = int(np.argmax(bites)) if bites.any() else len(seq)
        assert np.array_equal(mine[:first], plain[:first])
        assert may_differ or not bites.any()
        print(f"C {cap}: rule bites on {int(bites.sum())} tokens; misses exclusive {mine.sum()} plain {plain.sum()}")


# -- 3. fills -------------------------------------------------------------------------
def _file_slice(path: Path, off: int, n: int) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(off)
        return np.frombuffer(f.read(n), dtype=np.uint8)


@pytest.mark.parametrize("threads", [1, 8])
def test_reader_reads_exact_ranges_coalesced_or_not(gguf, threads):
    reader = es.ExpertReader(gguf, threads, mincore=True)
    rng = np.random.default_rng(threads)
    size = os.path.getsize(gguf)
    jobs, want = [], []
    for _ in range(12):
        off = int(rng.integers(0, size - 3_000_000))
        n = int(rng.integers(1, 2_000_000))
        jobs.append((off, n, bytearray(n)))
    jobs.append((jobs[0][0] + jobs[0][1], 1000, bytearray(1000)))        # adjacent: one run with job 0
    jobs.append((0, es.ALIGN, bytearray(es.ALIGN)))                         # aligned both ends
    jobs.append((size - 777, 777, bytearray(777)))                         # the file's last bytes
    st = reader.read(jobs)
    for off, n, buf in jobs:
        assert bytes(buf) == _file_slice(gguf, off, n).tobytes()
    assert st["bytes"] == sum(j[1] for j in jobs) and st["read_bytes"] >= st["bytes"]
    assert st["labeled"] and st["cold_reads"] + st["cached_reads"] == st["reads"]
    assert st["cold_bytes"] + st["cached_bytes"] >= st["bytes"]
    with pytest.raises(ValueError):
        reader.read([(size - 10, 20, bytearray(20))])
    reader.close()


@pytest.mark.parametrize("threads", [1, 8])
def test_fill_puts_the_files_bytes_or_their_repack_in_the_slot(gguf, threads):
    gx = experts_of(gguf)
    reader = es.ExpertReader(gx.path, threads)
    for layer, plans in enumerate(es.plans_from(gx)):
        arena = es.LayerArena(layer, plans, SLOTS)
        ids = np.array([3, 4, 5, 11, 0, 23, 17, 9, 10, 2])                 # adjacent runs and singles
        slots, misses = arena.plan(ids)
        arena.fill(misses, reader)
        for s, e in misses:
            for proj, plan in plans.items():
                raw = _file_slice(gx.path, plan.offset + e * plan.file_bytes, plan.file_bytes)
                if plan.codec is not None:
                    assert np.array_equal(arena.bufs[proj]["weight"].slot(s, plan.file_bytes), raw)
                else:
                    codes, scales, biases = q2_0_to_affine(raw.reshape(1, plan.rows, -1), plan.k)
                    for name, want in (("weight", codes), ("scales", scales), ("biases", biases)):
                        got = arena.bufs[proj][name].slot(s, want[0].size)
                        assert np.array_equal(got, want[0].reshape(-1).view(got.dtype))
    reader.close()


def test_a_failed_fill_leaves_its_slots_empty(gguf):
    gx = experts_of(gguf)
    arena = es.LayerArena(0, es.plans_from(gx)[0], SLOTS)

    class Broken:
        def read(self, jobs):
            raise OSError("short read")

    slots, misses = arena.plan(np.arange(10))
    with pytest.raises(OSError):
        arena.fill(misses, Broken())
    assert (arena.slot_of[:10] == -1).all() and (arena.expert_of == -1).all()


# -- 4-5. the arena path is exact -----------------------------------------------------
def _layer_stream(gx: GGUFExperts, layer: int, **kw) -> es.ExpertStream:
    return es.ExpertStream(gx, cfg(**kw), layers=[layer])


@pytest.mark.parametrize("layer,pin", [(i, False) for i in range(len(LAYERS))] + [(0, True), (1, True)])
def test_decode_calls_are_bit_identical_with_misses_and_evictions(gguf, stock, layer, pin):
    gx = experts_of(gguf)
    ref = make_switch(gx, layer, stock, E)
    stream = _layer_stream(gx, layer, pin=pin)
    assert (stream.pin is not None) == pin
    moe = FakeMoE(make_switch(gx, layer, None, SLOTS))
    hook = stream.layer(layer)
    rng = np.random.default_rng(100 + layer)
    for t in range(50):
        x, idx = x_of(rng, 1), route(rng, 1)
        want = ref(x, mx.array(idx))
        got = hook(moe, x, mx.array(idx))
        hook.done(got)
        assert np.array_equal(bits(want), bits(got)), (layer, t)
    calls = stream.drain()
    assert all(c["path"] == "arena" for c in calls)
    assert sum(c["misses"] for c in calls) > 2 * SLOTS                   # misses and evictions both happened
    assert all(0 <= c["settle_ms"] <= c["wait_ms"] and c["sync_ms"] <= c["wait_ms"] for c in calls)
    assert stream.check_views(_Model(layer, moe)) == (3 if gx.codec(layer, "down_proj") else 5)
    stream.close()


class _Model:
    def __init__(self, layer, moe):
        self.layers = {layer: type("L", (), {"mlp": moe})()}


@pytest.mark.parametrize("layer", [0, 1])
def test_multi_row_calls_take_the_arena_below_the_sort_threshold_and_the_ring_above(gguf, stock, layer):
    gx = experts_of(gguf)
    ref = make_switch(gx, layer, stock, E)
    stream = _layer_stream(gx, layer)
    moe = FakeMoE(make_switch(gx, layer, None, SLOTS))
    hook = stream.layer(layer)
    rng = np.random.default_rng(200 + layer)
    for rows in (2, 6, 6, 7, 3, 30, 1):
        pool = rng.choice(E, SLOTS, replace=False) if rows <= 6 else None
        x, idx = x_of(rng, rows), route(rng, rows, pool)
        got = hook(moe, x, mx.array(idx))
        hook.done(got)
        assert np.array_equal(bits(ref(x, mx.array(idx))), bits(got)), rows
        assert stream.calls[-1]["path"] == ("arena" if rows * TOP < es.SORT_MIN else "ring")
    stream.close()


@pytest.mark.skipif(os.environ.get("TF_STREAM_SUBRUN") == "1", reason="the subprocess itself")
def test_split_k_forced_on_is_still_exact(gguf):
    # kquant reads KQ_QMM_SPLITK once per process: the decode and wide-call tests again in a child with it set
    env = dict(os.environ, KQ_QMM_SPLITK="1", TF_STREAM_SUBRUN="1")
    here = Path(__file__).resolve()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(here), "-k",
                        "bit_identical_with_misses or sort_threshold or ring_stages"], env=env, cwd=here.parent.parent,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


# -- 6. the prefill ring --------------------------------------------------------------
@pytest.mark.parametrize("layer", [0, 1])
def test_ring_stages_prompt_chunks_exactly_and_warms_only_on_the_final_chunk(gguf, stock, layer):
    gx = experts_of(gguf)
    ref = make_switch(gx, layer, stock, E)
    stream = _layer_stream(gx, layer, prefill_rows=2048)
    for buf in stream.ring.bufs.values():                                # stale bytes everywhere in the ring
        buf.host[:] = np.random.default_rng(1).integers(0, 2**15, size=buf.host.size).astype(buf.host.dtype)
    moe = FakeMoE(make_switch(gx, layer, None, SLOTS))
    hook = stream.layer(layer)
    rng = np.random.default_rng(300 + layer)
    # some residents first (decode calls), so the ring copies hits from the arena and reads the rest
    for _ in range(4):
        x, idx = x_of(rng, 1), route(rng, 1)
        hook.done(hook(moe, x, mx.array(idx)))
    sizes = [b.nbytes for b in stream.ring.bufs.values()]
    held = stream.arenas[0].slot_of.copy()
    for rows, final in ((2048, False), (600, None)):
        stream.final_chunk = final
        x = x_of(rng, rows)
        idx = route(rng, rows, rng.choice(E, 20, replace=False))       # 20 of 24 experts: 4 positions stay stale
        got = hook(moe, x, mx.array(idx))
        hook.done(got)
        assert np.array_equal(bits(ref(x, mx.array(idx))), bits(got)), rows
        rec = stream.calls[-1]
        assert rec["path"] == "ring" and rec["copied"] > 0 and rec["misses"] > 0
        if final is False:
            assert "warmed" not in rec and np.array_equal(stream.arenas[0].slot_of, held)   # not evicted by prefill
        else:
            assert rec["warmed"] >= 0
            tail = set(np.asarray(idx)[0, -64:].reshape(-1).tolist())
            newest = [int(e) for e in np.asarray(idx)[0, -2:].reshape(-1)]
            assert all(stream.arenas[0].slot_of[e] >= 0 for e in newest) and len(tail) >= 1
    assert [b.nbytes for b in stream.ring.bufs.values()] == sizes          # nothing grows across calls
    # the warmed slots hold the right bytes: a decode call that hits them is exact and reads nothing
    resident = stream.arenas[0].resident()
    idx = np.array(rng.choice(resident, TOP, replace=False), dtype=np.uint32)[None, None]
    x = x_of(rng, 1)
    got = hook(moe, x, mx.array(idx))
    hook.done(got)
    assert np.array_equal(bits(ref(x, mx.array(idx))), bits(got))
    assert stream.calls[-1]["misses"] == 0
    stream.close()


# -- 7. the fence ---------------------------------------------------------------------
def _primed(gguf, stock, fence: bool):
    gx = experts_of(gguf)
    ref = make_switch(gx, 0, stock, E)
    stream = _layer_stream(gx, 0)
    stream.fence = fence
    moe = FakeMoE(make_switch(gx, 0, None, SLOTS))
    hook = stream.layer(0)
    rng = np.random.default_rng(5)
    for chunk in (np.arange(0, 10), np.arange(10, 16).tolist() + [0, 1, 2, 3]):    # 16 resident
        out = hook(moe, x_of(rng, 1), mx.array(np.array(chunk, dtype=np.uint32)[None, None]))
        hook.done(out)
        mx.eval(out)
    return ref, stream, moe, hook, rng


@pytest.mark.parametrize("fence", [True, False])
def test_a_lazy_call_survives_a_later_refill_only_with_the_fence(gguf, stock, fence):
    ref, stream, moe, hook, rng = _primed(gguf, stock, fence)
    a_ids = mx.array(np.arange(0, 10, dtype=np.uint32)[None, None])      # all hits, built lazily, never evaluated
    b_ids = mx.array(np.arange(14, 24, dtype=np.uint32)[None, None])     # 8 misses: evict slots A reads
    xa, xb = x_of(rng, 1), x_of(rng, 1)
    a = hook(moe, xa, a_ids)
    hook.done(a)
    b = hook(moe, xb, b_ids)                                             # the production fence only
    hook.done(b)
    assert stream.calls[-1]["misses"] == 8
    same_a = np.array_equal(bits(ref(xa, a_ids)), bits(a))
    assert np.array_equal(bits(ref(xb, b_ids)), bits(b))
    assert same_a == fence                                               # without it the hazard is real
    stream.close()


# -- guards ---------------------------------------------------------------------------
def test_batched_calls_and_interleaved_forwards_raise(gguf):
    import threading

    gx = experts_of(gguf, layers=2)
    stream = es.ExpertStream(gx, cfg())
    rng = np.random.default_rng(6)
    moe0, moe1 = (FakeMoE(make_switch(gx, i, None, SLOTS)) for i in (0, 1))
    with pytest.raises(RuntimeError, match="batch"):
        stream.layer(0)(moe0, mx.zeros((2, 1, K), dtype=mx.bfloat16), mx.array(route(rng, 1).repeat(2, 0)))
    stream.layer(0).done(stream.layer(0)(moe0, x_of(rng, 1), mx.array(route(rng, 1))))
    stream.layer(1).done(stream.layer(1)(moe1, x_of(rng, 1), mx.array(route(rng, 1))))
    with pytest.raises(RuntimeError, match="out of order"):
        stream.layer(1)(moe1, x_of(rng, 1), mx.array(route(rng, 1)))       # a second forward interleaved
    stream.layer(0).done(stream.layer(0)(moe0, x_of(rng, 1), mx.array(route(rng, 1))))
    errors = []

    def other() -> None:
        try:
            stream.layer(1)(moe1, x_of(rng, 1), mx.array(route(rng, 1)))
        except RuntimeError as err:
            errors.append(err)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert errors and "another thread" in str(errors[0])
    stream.close()


def test_settings_refusals_and_the_load_line(gguf):
    assert es.settings({}) is None and es.settings({"TF_EXPERT_STREAM": "0"}) is None
    with pytest.raises(ValueError):
        es.settings({"TF_EXPERT_STREAM": "1"})
    with pytest.raises(ValueError):
        es.settings({"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16", "TF_EXPERT_ARENA_GIB": "1"})
    s = es.settings({"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_GIB": "0.5", "TF_EXPERT_MINCORE": "1"})
    assert s["gib"] == 0.5 and s["slots"] is None and s["threads"] == 8 and s["mincore"]
    config = {"experts_source": {"format": "gguf-kquant", "file": "x.gguf"}}
    good = es.settings({"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16"})
    es.refuse(good, config, {})
    for env in ({"TF_WIRED_GIB": "30"}, {"TF_FLASH_REF_DRAFT": "pad"}, {"TF_FLASH_MTP_GGUF": "/h.gguf"}):
        with pytest.raises(ValueError):
            es.refuse(good, config, env)
    with pytest.raises(ValueError):
        es.refuse(good, {}, {})
    with pytest.raises(ValueError):
        es.refuse(es.settings({"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "8"}), config, {})
    gx = experts_of(gguf)
    plans = es.plans_from(gx)
    one = es.arena_bytes(plans, 1)
    assert es.slots_for_gib(plans, 17.5 * one / 2**30) == 17
    with pytest.raises(ValueError):
        es.ExpertStream(gx, cfg(slots=8), layers=[0])
    stream = es.ExpertStream(gx, cfg(slots=16, threads=3), layers=[0, 1])
    line = stream.summary()
    assert line.startswith("tensorfold: expert stream: 16 slots a layer (") and "io threads 3, mincore off" in line
    assert line.endswith(", unpinned") and stream.pin is None
    assert stream.calls.maxlen == es.CALLS_KEPT                          # per-call records stay bounded
    stream.close()
    assert not s["pin"] and es.settings({"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16",
                                         "TF_EXPERT_PIN": "1"})["pin"]


# -- pinning (TF_EXPERT_PIN=1) ------------------------------------------------------------
class _Calls:
    """``es._libc`` with mlock / munlock / minherit counted (and an mlock that fails from call ``fail_at``)."""

    def __init__(self, libc, fail_at: int | None = None):
        self._libc, self.fail_at, self.log = libc, fail_at, []

    def __getattr__(self, name):
        return getattr(self._libc, name)

    def _call(self, name, *args):
        self.log.append((name, int(args[0].value), int(args[1]), *args[2:]))
        if name == "mlock" and self.fail_at is not None and sum(c[0] == "mlock" for c in self.log) >= self.fail_at:
            ctypes.set_errno(12)
            return -1
        return getattr(self._libc, name)(*args)

    def mlock(self, *a):
        return self._call("mlock", *a)

    def munlock(self, *a):
        return self._call("munlock", *a)

    def minherit(self, *a):
        return self._call("minherit", *a)


def _all_bufs(stream):
    return [b for a in stream.arenas for p in a.bufs.values() for b in p.values()] + list(stream.ring.bufs.values())


def _child_reads(addr: int) -> int:
    """Fork; the child reads one byte at ``addr`` and exits 0. Returns the child's wait status."""

    pid = os.fork()
    if pid == 0:
        try:
            import faulthandler

            faulthandler.disable()                                       # die quietly on the expected fault
            ctypes.string_at(addr, 1)
        finally:
            os._exit(0)
    return os.waitpid(pid, 0)[1]


def test_pin_locks_every_buffer_keeps_it_out_of_forks_and_releases_on_close(gguf, monkeypatch):
    calls = _Calls(es._libc)
    monkeypatch.setattr(es, "_libc", calls)
    erased = []
    real_erase = kq.residency_erase
    monkeypatch.setattr(kq, "residency_erase", lambda a: erased.append(a) or real_erase(a))
    gx = experts_of(gguf, layers=2)
    stream = es.ExpertStream(gx, cfg(pin=True))
    bufs = _all_bufs(stream)
    spans = sorted((b.base, (b.nbytes + es.PAGE - 1) // es.PAGE * es.PAGE) for b in bufs)
    assert sorted(c[1:3] for c in calls.log if c[0] == "mlock") == spans
    assert sorted(c[1:3] for c in calls.log if c[0] == "minherit" and c[3] == es.VM_INHERIT_NONE) == spans
    assert stream.pin.nbytes == sum(n for _, n in spans) and stream.summary().endswith(
        f"pinned {stream.pin.nbytes / 2**30:.2f} GiB")
    assert all(_resident(b) for b in bufs)                               # mlock faulted every page in
    assert os.WIFSIGNALED(_child_reads(bufs[0].base))                    # pinned pages stay out of forks too
    stream.close()
    assert sorted(c[1:3] for c in calls.log if c[0] == "munlock") == spans and len(erased) == len(bufs)
    assert not stream.pin.pinned


@pytest.mark.parametrize("pin", [False, True])
def test_arena_and_ring_are_absent_from_a_forked_child(gguf, pin):
    gx = experts_of(gguf, layers=2)
    stream = es.ExpertStream(gx, cfg(pin=pin))
    for buf in (stream.arenas[0].bufs["gate_proj"]["weight"], next(iter(stream.ring.bufs.values()))):
        status = _child_reads(buf.base)
        assert os.WIFSIGNALED(status) and os.WTERMSIG(status) in (10, 11), status   # SIGBUS / SIGSEGV
    raw, mv = kq.arena_alloc([es.PAGE], 1)
    plain = np.frombuffer(mv, dtype=np.uint8)
    plain[:] = 1
    status = _child_reads(int(plain.__array_interface__["data"][0]))
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0          # control: a plain arena_alloc is inherited
    del plain, mv, raw
    stream.close()


def _resident(buf) -> bool:
    n = (buf.nbytes + es.PAGE - 1) // es.PAGE
    vec = ctypes.create_string_buffer(n)
    assert es._libc.mincore(ctypes.c_void_p(buf.base), n * es.PAGE, vec) == 0
    return bool((np.frombuffer(vec.raw, dtype=np.uint8) & 1).all())


def test_a_failed_pin_releases_what_it_pinned_and_runs_unpinned(gguf, monkeypatch, capsys):
    calls = _Calls(es._libc, fail_at=3)
    monkeypatch.setattr(es, "_libc", calls)
    gx = experts_of(gguf, layers=2)
    stream = es.ExpertStream(gx, cfg(pin=True))                      # mlock refused: the load still succeeds
    assert stream.pin is None and "mlock" in stream.pin_refused
    assert "not pinned, running unpinned" in capsys.readouterr().out and "(pin refused)" in stream.summary()
    locked = [c[1:3] for c in calls.log if c[0] == "mlock"]
    assert len(locked) == 3 and sorted(c[1:3] for c in calls.log if c[0] == "munlock") == sorted(locked[:2])
    stream.close()
    calls.log.clear()
    stream = es.ExpertStream(gx, cfg())
    with pytest.raises(ValueError, match="cap"):
        es.Pin(_all_bufs(stream), max_bytes=es.PAGE)
    assert not [c for c in calls.log if c[0] == "mlock"]                 # refused before anything was locked
    stream.close()


def test_a_pin_over_the_machines_cap_runs_unpinned_without_locking(gguf, monkeypatch):
    calls = _Calls(es._libc)
    monkeypatch.setattr(es, "_libc", calls)
    monkeypatch.setattr(es, "pin_cap", lambda memsize=None: float(es.PAGE))
    stream = es.ExpertStream(experts_of(gguf, layers=2), cfg(pin=True))
    assert stream.pin is None and "cap" in stream.pin_refused
    assert not [c for c in calls.log if c[0] == "mlock"]
    stream.close()


def test_a_refused_residency_set_runs_unpinned_and_unlocks(gguf, monkeypatch):
    calls = _Calls(es._libc)
    monkeypatch.setattr(es, "_libc", calls)
    monkeypatch.setattr(kq, "residency_commit", lambda: False)
    stream = es.ExpertStream(experts_of(gguf, layers=2), cfg(pin=True))
    assert stream.pin is None and "residency_commit" in stream.pin_refused
    locked = sorted(c[1:3] for c in calls.log if c[0] == "mlock")
    assert locked and sorted(c[1:3] for c in calls.log if c[0] == "munlock") == locked
    stream.close()


def test_pin_cap_scales_with_memory_and_keeps_the_ceiling():
    gib = 2**30
    assert es.pin_cap(48 * gib) == es.PIN_MAX_GIB * gib                  # the ceiling on a 48 GiB machine
    assert es.pin_cap(128 * gib) == es.PIN_MAX_GIB * gib
    assert es.pin_cap(16 * gib) == 16 * gib * es.PIN_FRACTION
    assert es.pin_cap(8 * gib) == 3.5 * gib
    assert es.pin_cap(0) == es.PIN_MAX_GIB * gib                         # unknown memory: the ceiling
    assert 0 < es.pin_cap() <= es.PIN_MAX_GIB * gib


@pytest.mark.parametrize("layer", range(len(LAYERS)))
def test_the_fixture_gives_finite_distinct_outputs(gguf, stock, layer):
    gx = experts_of(gguf)
    ref = make_switch(gx, layer, stock, E)
    rng = np.random.default_rng(9)
    x = x_of(rng, 1)
    y = ref(x, mx.array(np.arange(E, dtype=np.uint32).reshape(1, 1, -1)[..., :TOP]))
    z = ref(x, mx.array(np.arange(E, dtype=np.uint32).reshape(1, 1, -1)[..., TOP:2 * TOP]))
    yf, zf = np.array(y.astype(mx.float32)), np.array(z.astype(mx.float32))
    assert np.isfinite(yf).all() and np.abs(yf).max() > 0
    assert not np.array_equal(yf, zf)                     # experts differ, so a wrong slot shows


# -- 8. a tiny model end to end through model.load ------------------------------------
TINY_LAYERS = [("IQ4_NL", "IQ4_NL"), ("IQ2_S", "Q2_0"), ("IQ4_XS", "IQ4_NL"), ("IQ3_S", "Q2_0")]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory) -> Path:
    import json

    from mlx.utils import tree_flatten

    from tensorfold.families.qwen4_exp import model as q4
    from tests.test_qwen4_exp_family import TEXT

    text = dict(TEXT, hidden_size=256, moe_intermediate_size=256, num_experts=32, num_experts_per_tok=4,
                shared_expert_intermediate_size=64, linear_key_head_dim=32, linear_value_head_dim=32)
    d = tmp_path_factory.mktemp("tiny")
    mx.random.seed(11)
    model = q4.Qwen4Exp(q4.Config.from_dict({"text_config": text}))
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        if ".switch_mlp." in name:
            continue
        if name.endswith("norm.weight") or "layernorm" in name or "hc_norm" in name or name.endswith("norm_key.weight"):
            value = 0.1 * mx.random.normal(value.shape)
        weights[f"language_model.{name}"] = value.astype(mx.bfloat16)          # bf16 activations, as the model
    mx.save_safetensors(str(d / "model.safetensors"), weights)
    rng = np.random.default_rng(12)
    keep = {str(i): sorted(rng.choice(32, E, replace=False).tolist()) for i in range(len(TINY_LAYERS))}
    (d / "cut.json").write_text(json.dumps({"keep_local": keep}))
    write_gguf(d / "experts.gguf", expert_tensors(rng, TINY_LAYERS, 256, 256, E))
    (d / "config.json").write_text(json.dumps({"text_config": text, "eos_token_id": 5,
                                               "experts_source": {"format": "gguf-kquant", "file": "experts.gguf"},
                                               "experts_cut": {"file": "cut.json"}}))
    return d


def _tiny_run(tiny_dir: Path, monkeypatch, env: dict[str, str], prompt: list[int], chunk: int, steps: int):
    import mlx_lm.utils

    from tensorfold.families.qwen4_exp import model as q4

    for key in [k for k in os.environ if k.startswith("TF_EXPERT_")]:
        monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", lambda *a, **k: None)
    model, _ = q4.load(tiny_dir)
    stream = model.__dict__.get("expert_stream")
    cache = model.make_cache()
    steps_bits, tokens = [], []
    for begin in range(0, len(prompt), chunk):
        out = model(np.array([prompt[begin:begin + chunk]]), cache)
        mx.eval(out)
    for _ in range(steps):
        last = out[0, -1]
        steps_bits.append(bits(last))
        tokens.append(int(mx.argmax(last).item()))
        out = model(np.array([[tokens[-1]]]), cache)
    totals = dict(stream.totals) if stream is not None else None
    if stream is not None:
        stream.close()
    return steps_bits, tokens, totals, model


@pytest.mark.parametrize("prompt_len, chunk, prefill_rows, pin", [(23, 23, 2048, False), (80, 32, 32, True)])
def test_tiny_model_streamed_equals_resident_logits_bit_for_bit(tiny_dir, monkeypatch, capsys, prompt_len, chunk,
                                                                 prefill_rows, pin):
    rng = np.random.default_rng(prompt_len)
    prompt = [int(t) for t in rng.integers(6, 97, size=prompt_len)]
    base_bits, base_tokens, none, base_model = _tiny_run(tiny_dir, monkeypatch, {}, prompt, chunk, 40)
    assert none is None and "expert_stream" not in base_model.__dict__
    assert base_model.__dict__["gguf_experts"].ranges()                       # the stock route mapped the file
    env = {"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16", "TF_EXPERT_PREFILL_ROWS": str(prefill_rows),
           "TF_EXPERT_DEBUG": "1", "TF_EXPERT_MINCORE": "1", "TF_EXPERT_PIN": "1" if pin else "0"}
    got_bits, got_tokens, totals, model = _tiny_run(tiny_dir, monkeypatch, env, prompt, chunk, 40)
    out = capsys.readouterr().out
    assert "tensorfold: expert stream: 16 slots a layer (" in out and ("mincore on, pinned " in out) == pin
    assert (model.__dict__["expert_stream"].pin is not None) == pin
    assert model.__dict__["gguf_experts"].ranges() == []
    assert len(set(base_tokens)) > 3 and all(np.isfinite((b.astype(np.uint32) << 16).view(np.float32)).all() for b in base_bits)
    assert got_tokens == base_tokens
    for step, (a, b) in enumerate(zip(base_bits, got_bits)):
        assert np.array_equal(a, b), step
    assert totals["misses"] > 0 and totals["calls_arena"] > 0 and totals["calls_ring"] > 0
    assert totals.get("warmed", 0) > 0 and totals["cold_reads"] + totals["cached_reads"] == totals["reads"]


def test_tiny_model_refuses_streaming_with_a_wired_limit_or_drafting(tiny_dir, monkeypatch):
    import mlx_lm.utils

    from tensorfold.families.qwen4_exp import model as q4

    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setenv("TF_EXPERT_STREAM", "1")
    monkeypatch.setenv("TF_EXPERT_ARENA_SLOTS", "16")
    for key, value in (("TF_WIRED_GIB", "20"), ("TF_FLASH_REF_DRAFT", "pad")):
        monkeypatch.setenv(key, value)
        with pytest.raises(ValueError):
            q4.load(tiny_dir)
        monkeypatch.delenv(key)
    monkeypatch.setenv("TF_EXPERT_ARENA_SLOTS", "12")
    with pytest.raises(ValueError):
        q4.load(tiny_dir)


def _tiny_load(tiny_dir: Path, monkeypatch, env: dict[str, str]):
    import mlx_lm.utils

    from tensorfold.families.qwen4_exp import model as q4

    for key in [k for k in os.environ if k.startswith("TF_EXPERT_")]:
        monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", lambda *a, **k: None)
    return q4.load(tiny_dir)[0]


def _tiny_request(model, prompt: list[int], chunk: int, steps: int) -> list[np.ndarray]:
    """One request as the server runs it: a fresh cache, the prompt in chunks, then greedy decode steps."""

    cache = model.make_cache()
    for begin in range(0, len(prompt), chunk):
        out = model(np.array([prompt[begin:begin + chunk]]), cache)
        mx.eval(out)
    got = []
    for _ in range(steps):
        last = out[0, -1]
        got.append(bits(last))
        out = model(np.array([[int(mx.argmax(last).item())]]), cache)
        mx.eval(out)
    return got


def test_tiny_model_streams_requests_from_new_threads_bit_for_bit(tiny_dir, monkeypatch):
    """The server loads in the main thread and runs each request in a new thread (ThreadingHTTPServer): MLX's
    streams are per thread, so nothing the stream builds at load may stay lazy on the main thread's stream."""

    import threading

    rng = np.random.default_rng(31)
    prompts = [[int(t) for t in rng.integers(6, 97, size=n)] for n in (80, 47)]
    base_model = _tiny_load(tiny_dir, monkeypatch, {})
    base = [_tiny_request(base_model, p, 32, 12) for p in prompts]
    env = {"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16", "TF_EXPERT_PREFILL_ROWS": "32",
           "TF_EXPERT_DEBUG": "1"}
    model = _tiny_load(tiny_dir, monkeypatch, env)
    stream = model.__dict__["expert_stream"]
    got: list[list[np.ndarray] | BaseException] = []

    def request(prompt: list[int]) -> None:
        try:
            got.append(_tiny_request(model, prompt, 32, 12))
        except BaseException as err:                                       # noqa: BLE001 - reported below
            got.append(err)

    for prompt in prompts:                                                 # one at a time, a new thread each
        t = threading.Thread(target=request, args=(prompt,))
        t.start()
        t.join()
    assert not [g for g in got if isinstance(g, BaseException)], got
    for want, have in zip(base, got):
        for step, (a, b) in enumerate(zip(want, have)):
            assert np.array_equal(a, b), step
    assert stream.totals["calls_ring"] > 0 and stream.totals["calls_arena"] > 0
    stream.close()


# -- 9. resident layers (TF_EXPERT_RESIDENT, TF_EXPERT_LAYER_SLOTS) --------------------
def test_resident_and_layer_slot_settings_refuse_bad_values(gguf):
    on = {"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16"}
    s = es.settings(on)
    assert s["resident"] == [] and s["layer_slots"] == {}
    s = es.settings(dict(on, TF_EXPERT_RESIDENT=" 3, 0 ", TF_EXPERT_LAYER_SLOTS="1:20, 2:16"))
    assert s["resident"] == [0, 3] and s["layer_slots"] == {1: 20, 2: 16}
    for env, match in ((dict(on, TF_EXPERT_RESIDENT="1,x"), "not a layer index"),
                       (dict(on, TF_EXPERT_RESIDENT="-1"), "negative"),
                       (dict(on, TF_EXPERT_RESIDENT="1,1"), "twice"),
                       (dict(on, TF_EXPERT_RESIDENT="1,,2"), "not a layer index"),
                       (dict(on, TF_EXPERT_LAYER_SLOTS="1"), "layer:slots"),
                       (dict(on, TF_EXPERT_LAYER_SLOTS="1:x"), "integer"),
                       (dict(on, TF_EXPERT_LAYER_SLOTS="1:8"), "at least 16"),
                       (dict(on, TF_EXPERT_LAYER_SLOTS="1:20,1:24"), "twice"),
                       (dict(on, TF_EXPERT_RESIDENT="0,1", TF_EXPERT_LAYER_SLOTS="1:20"), "both name")):
        with pytest.raises(ValueError, match=match):
            es.settings(env)
    gx = experts_of(gguf)
    for kw, match in (({"resident": [4]}, "not streamed layers"), ({"layer_slots": {7: 20}}, "not streamed layers"),
                      ({"resident": [1], "layer_slots": {1: 20}}, "same layer"),
                      ({"layer_slots": {1: 8}}, "at least 16")):
        with pytest.raises(ValueError, match=match):
            es.ExpertStream(gx, cfg(**kw))
    with pytest.raises(ValueError, match="not streamed layers"):
        es.ExpertStream(gx, cfg(resident=[0]), layers=[1])                  # model layer ids, not positions


def test_resident_arena_holds_the_files_bytes_for_every_expert(gguf):
    gx = experts_of(gguf)
    stream = es.ExpertStream(gx, cfg(resident=[1, 2]))
    assert [a.C for a in stream.arenas] == [SLOTS, E, E, SLOTS] and [a.fixed for a in stream.arenas] == [False, True, True, False]
    assert stream.resident_load["bytes"] > 0 and stream.resident_bytes == stream.arenas[1].nbytes + stream.arenas[2].nbytes
    for arena in stream.arenas[1:3]:
        assert np.array_equal(arena.slot_of, np.arange(E)) and np.array_equal(arena.expert_of, np.arange(E))
        for proj, plan in arena.plans.items():
            raw = _file_slice(gx.path, plan.offset, E * plan.file_bytes).reshape(E, plan.file_bytes)
            if plan.codec is not None:
                assert np.array_equal(arena.bufs[proj]["weight"].host.reshape(E, -1), raw)
            else:
                codes, scales, biases = q2_0_to_affine(raw.reshape(E, plan.rows, -1), plan.k)
                for name, want in (("weight", codes), ("scales", scales), ("biases", biases)):
                    got = arena.bufs[proj][name].host
                    assert np.array_equal(got, want.reshape(-1).view(got.dtype)), (proj, name)
        for call in (lambda a: a.plan(np.arange(3)), lambda a: a.touch(np.arange(3)), lambda a: a.assign([0]),
                     lambda a: a.fill([(0, 0)], stream.reader)):
            with pytest.raises(RuntimeError, match="resident arena"):
                call(arena)
    # the ring is sized for the streamed layers only (layers 0 and 3 here)
    ring = es.PrefillRing([stream.arenas[0], stream.arenas[3]])
    assert stream.ring.nbytes == ring.nbytes
    stream.close()


class _HostSyncs:
    """``np.array`` on an MLX array counted (the stream's host sync on the routed ids)."""

    def __init__(self, monkeypatch):
        self.n, real = 0, np.array

        def counted(obj, *a, **k):
            if isinstance(obj, mx.array):
                self.n += 1
            return real(obj, *a, **k)

        monkeypatch.setattr(es.np, "array", counted)


@pytest.mark.parametrize("ring", ["split", "shared"])
def test_resident_layers_take_the_stock_call_without_a_host_sync(gguf, stock, monkeypatch, ring):
    gx = experts_of(gguf)
    stream = es.ExpertStream(gx, cfg(resident=[0, 1], layer_slots={2: 20}, ring=ring))
    assert [a.C for a in stream.arenas] == [E, E, 20, SLOTS]
    refs = [make_switch(gx, i, stock, E) for i in range(len(LAYERS))]
    moes = [FakeMoE(make_switch(gx, i, None, stream.arenas[i].C)) for i in range(len(LAYERS))]
    hooks = [stream.layer(i) for i in range(len(LAYERS))]
    assert [type(h).__name__ for h in hooks] == ["ResidentLayer", "ResidentLayer", "LayerStream", "LayerStream"]
    reads = []
    real_read = stream.reader.read
    monkeypatch.setattr(stream.reader, "read", lambda jobs: reads.append(len(jobs)) or real_read(jobs))
    syncs = _HostSyncs(monkeypatch)
    rng = np.random.default_rng(41)
    for step, rows in enumerate((1, 1, 6, 30, 1, 2, 1)):                  # decode, short and ring-sized calls
        for i, (ref, moe, hook) in enumerate(zip(refs, moes, hooks)):
            x, idx = x_of(rng, rows), route(rng, rows)
            before, n_reads = syncs.n, len(reads)
            got = hook(moe, x, mx.array(idx))
            hook.done(got)
            if i < 2:
                assert syncs.n == before and len(reads) == n_reads, (step, i)   # no host sync, no reads
            else:
                assert syncs.n > before
            assert np.array_equal(bits(ref(x, mx.array(idx))), bits(got)), (step, i, rows)
    calls = stream.drain()
    assert {c["layer"] for c in calls} == {2, 3} and len(calls) == 2 * 7  # no records for resident layers
    assert stream.resident_calls == 2 * 7
    assert all(stream.arenas[i].clock == 0 and np.array_equal(stream.arenas[i].slot_of, np.arange(E)) for i in (0, 1))
    line = stream.summary()
    assert f"resident layers [0, 1] ({stream.resident_bytes / 2**30:.2f} GiB), layer slots 2:20, ring " in line
    # the order and thread checks still see resident layers: a skipped or repeated layer raises
    with pytest.raises(RuntimeError, match="out of order"):
        hooks[1](moes[1], x_of(rng, 1), mx.array(route(rng, 1)))
    hooks[0](moes[0], x_of(rng, 1), mx.array(route(rng, 1)))
    with pytest.raises(RuntimeError, match="out of order"):
        hooks[2](moes[2], x_of(rng, 1), mx.array(route(rng, 1)))
    with pytest.raises(RuntimeError, match="batch"):
        hooks[0](moes[0], mx.zeros((2, 1, K), dtype=mx.bfloat16), mx.array(route(rng, 1).repeat(2, 0)))
    stream.close()


def test_arena_gib_pays_for_resident_and_overridden_layers_first(gguf):
    gx = experts_of(gguf)
    plans = es.plans_from(gx)
    fixed = es.arena_bytes([plans[0]], E) + es.arena_bytes([plans[1]], 20)
    gib = (fixed + es.arena_bytes(plans[2:], 18) + 1000) / 2**30
    stream = es.ExpertStream(gx, cfg(slots=None, gib=gib, resident=[0], layer_slots={1: 20}))
    assert stream.slots == 18 and [a.C for a in stream.arenas] == [E, 20, 18, 18]
    stream.close()
    with pytest.raises(ValueError, match="at least 16"):                  # the rest no longer fits 16 slots
        es.ExpertStream(gx, cfg(slots=None, gib=(fixed + es.arena_bytes(plans[2:], 15)) / 2**30, resident=[0],
                                layer_slots={1: 20}))
    stream = es.ExpertStream(gx, cfg(slots=None, gib=0.001, resident=[0, 1, 2, 3]))   # nothing left to size
    assert all(a.fixed for a in stream.arenas) and stream.ring.nbytes == 0
    stream.close()


def test_pin_covers_resident_arenas_and_releases_them_on_close(gguf, monkeypatch):
    calls = _Calls(es._libc)
    monkeypatch.setattr(es, "_libc", calls)
    gx = experts_of(gguf, layers=2)
    stream = es.ExpertStream(gx, cfg(pin=True, resident=[1]))
    bufs = _all_bufs(stream)
    resident = [b for p in stream.arenas[1].bufs.values() for b in p.values()]
    spans = sorted((b.base, (b.nbytes + es.PAGE - 1) // es.PAGE * es.PAGE) for b in bufs)
    assert all(b in bufs for b in resident)
    assert sorted(c[1:3] for c in calls.log if c[0] == "mlock") == spans
    assert stream.pin.nbytes == sum(n for _, n in spans) and stream.pin.nbytes > stream.resident_bytes
    assert all(_resident(b) for b in resident)
    stream.close()
    assert sorted(c[1:3] for c in calls.log if c[0] == "munlock") == spans and not stream.pin.pinned


@pytest.mark.parametrize("prompt_len, chunk, prefill_rows, pin, resident, layer_slots",
                         [(23, 23, 2048, False, "0,2", ""), (80, 32, 32, True, "1,3", ""),
                          (80, 32, 32, False, "0,1", "3:20")])
def test_tiny_model_with_resident_layers_equals_both_other_builds_bit_for_bit(tiny_dir, monkeypatch, capsys,
                                                                               prompt_len, chunk, prefill_rows, pin,
                                                                               resident, layer_slots):
    rng = np.random.default_rng(prompt_len)
    prompt = [int(t) for t in rng.integers(6, 97, size=prompt_len)]
    base_bits, base_tokens, _, _ = _tiny_run(tiny_dir, monkeypatch, {}, prompt, chunk, 40)
    env = {"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16", "TF_EXPERT_PREFILL_ROWS": str(prefill_rows),
           "TF_EXPERT_DEBUG": "1", "TF_EXPERT_PIN": "1" if pin else "0"}
    plain_bits, plain_tokens, plain_totals, _ = _tiny_run(tiny_dir, monkeypatch, env, prompt, chunk, 40)
    env.update(TF_EXPERT_RESIDENT=resident, TF_EXPERT_LAYER_SLOTS=layer_slots)
    got_bits, got_tokens, totals, model = _tiny_run(tiny_dir, monkeypatch, env, prompt, chunk, 40)
    out = capsys.readouterr().out
    stream = model.__dict__["expert_stream"]
    layers = [int(i) for i in resident.split(",")]
    assert f"resident layers {layers} (" in out and (f"layer slots {layer_slots.replace(',', ' ')}" in out
                                                     if layer_slots else "layer slots" not in out)
    assert got_tokens == base_tokens == plain_tokens and len(set(base_tokens)) > 3
    for step, (a, b, c) in enumerate(zip(base_bits, plain_bits, got_bits)):
        assert np.array_equal(a, c) and np.array_equal(b, c), step
    assert stream.resident_calls == len(layers) * (-(-prompt_len // chunk) + 40)
    assert totals["calls_arena"] + totals["calls_ring"] == (4 - len(layers)) * (-(-prompt_len // chunk) + 40)
    assert totals["calls_ring"] > 0 and totals["calls_arena"] > 0
    for i in layers:
        arena = stream.arenas[stream.index[i]]
        assert arena.fixed and arena.clock == 0 and arena.C == arena.E
        assert model.layers[i].mlp.switch_mlp.gate_proj["weight"].shape[0] == arena.E
    for item in filter(None, layer_slots.split(",")):
        i, c = map(int, item.split(":"))
        assert stream.arenas[stream.index[i]].C == c
