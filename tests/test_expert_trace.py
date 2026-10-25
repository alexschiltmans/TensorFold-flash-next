"""TF_EXPERT_TRACE: the router trace hook in expert_stream.py (format rtrace1).

CPU tests (no Metal): settings, the _Trace writer (round trip through an independent reader, exclusive create, close
semantics) and a microbenchmark of one decode record.

Metal tests (``metal_`` in the name; they need an Apple GPU, so the CPU pass deselects them): a tiny streamed model with the trace on
must give the trace-off logits bit for bit; one record per streamed layer call; each record's routed ids equal the ids the
layer routed; the miss count equals the call's ``misses``; and rtrace_sim's ExclusiveLRU reproduces every call.
"""

from __future__ import annotations

import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
kq = pytest.importorskip("mlx_kquant")

from tensorfold.families.qwen4_exp import expert_stream as es  # noqa: E402

SIM = Path(__file__).resolve().parents[2] / "rtrace" / "code"       # opt/rtrace/code (rtrace_sim.py)
RECORD = struct.Struct("<QIIBBBBHHHHIfff")                           # written independently of es._Trace.HEAD
K = 10


def _read_trace(path: Path):
    raw = path.read_bytes()
    nl = raw.index(b"\n")
    import json

    header = json.loads(raw[:nl])
    pos, out = nl + 1, []
    while pos < len(raw):
        fields = RECORD.unpack_from(raw, pos)
        rows, top_k, nmiss = fields[2], fields[6], fields[10]
        start = pos + RECORD.size
        ids = np.frombuffer(raw, np.uint8, rows * top_k, start).astype(np.int64)
        miss = np.frombuffer(raw, np.uint8, nmiss, start + rows * top_k).astype(np.int64)
        out.append((fields, ids, miss))
        pos = start + rows * top_k + nmiss
    assert pos == len(raw), "no partial record"
    return header, out


# ---- CPU ----------------------------------------------------------------------------------------------------------------
def test_record_head_is_44_bytes_as_its_fields_say():
    # u64 + 2 x u32 + 4 x u8 + 4 x u16 + u32 + 3 x f32
    assert RECORD.size == 44 == es._Trace.HEAD.size
    assert es._Trace.HEAD.format == "<QIIBBBBHHHHIfff"


def test_settings_trace_is_none_unset_and_the_path_when_set():
    base = {"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16"}
    assert es.settings(base)["trace"] is None
    assert es.settings({**base, "TF_EXPERT_TRACE": ""})["trace"] is None
    assert es.settings({**base, "TF_EXPERT_TRACE": " /tmp/x.rtr "})["trace"] == "/tmp/x.rtr"
    assert es.settings({"TF_EXPERT_STREAM": "0", "TF_EXPERT_TRACE": "/x"}) is None


def test_trace_round_trip_decode_six_row_and_ring_records(tmp_path):
    rng = np.random.default_rng(2)
    path = tmp_path / "t.rtr"
    header = {"format": "rtrace1", "layers": [{"layer": 5, "E": 40, "C": 16}]}
    tr = es._Trace(str(path), header)
    dec_ids = rng.integers(0, 40, size=K)
    six_ids = rng.integers(0, 40, size=6 * K)
    ring_ids = rng.integers(0, 40, size=2048 * K)
    dec_miss = np.array([3, 17], np.int64)
    six_miss = np.array([], np.int64)
    ring_miss = np.sort(rng.choice(40, size=30, replace=False)).astype(np.int64)
    tr.add(t_ns=111, fwd=0, rows=1, layer=5, path=0, flags=8, top_k=K, uniq=9, misses=2, warmed=0,
           read_bytes=2 * 2**20, read_ms=1.25, wait_ms=2.5, sync_ms=0.5, ids=dec_ids, miss_ids=dec_miss)
    tr.add(t_ns=222, fwd=1, rows=6, layer=5, path=0, flags=0, top_k=K, uniq=12, misses=0, warmed=0,
           read_bytes=0, read_ms=0.0, wait_ms=0.1, sync_ms=0.2, ids=six_ids, miss_ids=six_miss)
    tr.add(t_ns=333, fwd=2, rows=2048, layer=5, path=1, flags=1 | 4, top_k=K, uniq=30, misses=30, warmed=64,
           read_bytes=2**32 + 5, read_ms=9.5, wait_ms=12.0, sync_ms=0.75, ids=ring_ids, miss_ids=ring_miss)
    tr.close()
    header_back, recs = _read_trace(path)
    assert header_back == header
    assert len(recs) == 3
    (f0, i0, m0), (f1, i1, m1), (f2, i2, m2) = recs
    assert f0[:10] == (111, 0, 1, 5, 0, 8, K, 9, 2, 0) and f0[11] == 2 * 2**20
    assert f0[12] == pytest.approx(1.25) and f0[13] == pytest.approx(2.5) and f0[14] == pytest.approx(0.5)
    assert np.array_equal(i0, dec_ids) and np.array_equal(m0, dec_miss)
    assert f1[:10] == (222, 1, 6, 5, 0, 0, K, 12, 0, 0) and np.array_equal(i1, six_ids) and m1.size == 0
    assert f2[:10] == (333, 2, 2048, 5, 1, 5, K, 30, 30, 64)
    assert f2[11] == 2**32 - 1, "read_bytes clamps to u32"
    assert np.array_equal(i2, ring_ids) and np.array_equal(m2, ring_miss)


def test_an_existing_path_is_refused(tmp_path):
    p = tmp_path / "exists.rtr"
    p.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        es._Trace(str(p), {"format": "rtrace1"})
    assert p.read_bytes() == b"old"


def test_flush_and_close_are_idempotent_and_add_after_close_raises(tmp_path):
    p = tmp_path / "c.rtr"
    tr = es._Trace(str(p), {"format": "rtrace1"})
    tr.add(t_ns=1, fwd=0, rows=1, layer=0, path=0, flags=8, top_k=K, uniq=1, misses=0, warmed=0, read_bytes=0,
           read_ms=0.0, wait_ms=0.0, sync_ms=0.0, ids=np.arange(K), miss_ids=np.zeros(0, np.int64))
    tr.close()
    size = p.stat().st_size
    tr.flush()
    tr.close()
    assert p.stat().st_size == size
    with pytest.raises(RuntimeError):
        tr.add(t_ns=2, fwd=0, rows=1, layer=0, path=0, flags=8, top_k=K, uniq=1, misses=0, warmed=0, read_bytes=0,
               read_ms=0.0, wait_ms=0.0, sync_ms=0.0, ids=np.arange(K), miss_ids=np.zeros(0, np.int64))


def test_a_length_mismatch_is_refused(tmp_path):
    tr = es._Trace(str(tmp_path / "m.rtr"), {"format": "rtrace1"})
    with pytest.raises(ValueError):
        tr.add(t_ns=1, fwd=0, rows=2, layer=0, path=0, flags=0, top_k=K, uniq=1, misses=0, warmed=0, read_bytes=0,
               read_ms=0.0, wait_ms=0.0, sync_ms=0.0, ids=np.arange(K), miss_ids=np.zeros(0, np.int64))
    tr.close()


def test_decode_add_costs_at_most_20_microseconds_on_average(tmp_path):
    tr = es._Trace(str(tmp_path / "bench.rtr"), {"format": "rtrace1"})
    ids = np.arange(K, dtype=np.int64) * 3
    miss = np.array([3, 9], np.int64)
    n = 10_000
    t0 = time.perf_counter()
    for i in range(n):
        tr.add(t_ns=i, fwd=i, rows=1, layer=7, path=0, flags=8, top_k=K, uniq=10, misses=2, warmed=0,
               read_bytes=4096, read_ms=0.1, wait_ms=0.2, sync_ms=0.1, ids=ids, miss_ids=miss)
        if i % 48 == 47:
            tr.flush()
    dt = time.perf_counter() - t0
    tr.close()
    per_call_us = dt / n * 1e6
    print(f"decode add: {per_call_us:.2f} us a call (10,000 calls, flush every 48)")
    assert per_call_us <= 20.0


# ---- Metal (written, not run: needs the owner's go) -----------------------------------------------------------------
from tests.test_expert_stream import _tiny_run, tiny_dir  # noqa: E402,F401  (the tiny streamed model fixture)


@pytest.mark.parametrize("prompt_len, chunk", [(23, 23), (80, 32)])
def test_metal_trace_on_is_bit_identical_and_records_every_streamed_call(tiny_dir, monkeypatch, tmp_path, prompt_len,
                                                                       chunk):
    sys.path.insert(0, str(SIM))
    import rtrace_sim as rs

    rng = np.random.default_rng(prompt_len)
    prompt = [int(t) for t in rng.integers(6, 97, size=prompt_len)]
    base_bits, base_tokens, _, _ = _tiny_run(tiny_dir, monkeypatch, {}, prompt, chunk, 40)
    trace_path = tmp_path / "tiny.rtr"
    env = {"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16", "TF_EXPERT_PREFILL_ROWS": "2048",
           "TF_EXPERT_TRACE": str(trace_path)}

    captured: list[tuple[int, np.ndarray]] = []
    recs: list[dict] = []
    orig_call = es.LayerStream.__call__
    orig_record = es.ExpertStream.record

    def call_spy(self, moe, x, experts):
        captured.append((self.arena.layer, np.array(experts).reshape(-1).astype(np.int64)))
        return orig_call(self, moe, x, experts)

    def record_spy(self, rec):
        recs.append(dict(rec))
        return orig_record(self, rec)

    monkeypatch.setattr(es.LayerStream, "__call__", call_spy)
    monkeypatch.setattr(es.ExpertStream, "record", record_spy)
    got_bits, got_tokens, totals, model = _tiny_run(tiny_dir, monkeypatch, env, prompt, chunk, 40)
    assert got_tokens == base_tokens
    for a, b in zip(base_bits, got_bits):
        assert np.array_equal(a, b)

    header, recs_on_disk = _read_trace(trace_path)
    assert len(recs_on_disk) == len(recs) == len(captured) == totals["calls_arena"] + totals["calls_ring"]
    assert header["format"] == "rtrace1" and header["ring"] == "split"
    meta = {m["layer"]: m for m in header["layers"]}
    replay = {L: rs.ExclusiveLRU(m["E"], m["C"]) for L, m in meta.items()}
    for (fields, ids, miss), (layer, routed), rec in zip(recs_on_disk, captured, recs):
        assert layer == rec["layer"] == fields[3]
        assert np.array_equal(ids, routed), "the record holds the ids the layer routed"
        assert len(miss) == rec["misses"] == fields[8], "miss ids count equals the call's misses"
        pol = replay[layer]
        uniq = np.unique(ids)
        if fields[4] == 0:
            want = sorted(pol.plan(ids))
        else:
            pol.touch(ids)
            want = sorted(int(e) for e in uniq[pol.slot_of[uniq] < 0])
            warmed = pol.assign(rs.warm_order(ids, K, header["warm_rows"], pol.C)) if fields[5] & 1 else 0
            assert warmed == fields[9]
        assert np.array_equal(miss, np.asarray(want, np.int64)), "rtrace_sim's ExclusiveLRU reproduces the call"


@pytest.mark.parametrize("ring", ["split", "shared"])
def test_metal_ring_flag_follows_the_ring_kind(tiny_dir, monkeypatch, tmp_path, ring):
    trace_path = tmp_path / f"{ring}.rtr"
    env = {"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16", "TF_EXPERT_PREFILL_ROWS": "2048",
           "TF_EXPERT_TRACE": str(trace_path), "TF_EXPERT_RING": ring}
    prompt = [int(t) for t in np.random.default_rng(3).integers(6, 97, size=80)]
    _tiny_run(tiny_dir, monkeypatch, env, prompt, 32, 4)
    header, recs = _read_trace(trace_path)
    assert header["ring"] == ring
    ring_recs = [f for f, _, _ in recs if f[4] == 1]
    assert ring_recs, "the prompt takes the ring path"
    assert all(bool(f[5] & 4) == (ring == "shared") for f in ring_recs)
    assert all(bool(f[5] & 8) == (f[2] == 1) for f, _, _ in recs)


def test_metal_every_streamed_layer_has_records_in_every_forward(tiny_dir, monkeypatch, tmp_path):
    trace_path = tmp_path / "all.rtr"
    env = {"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16", "TF_EXPERT_PREFILL_ROWS": "2048",
           "TF_EXPERT_TRACE": str(trace_path)}
    prompt = [int(t) for t in np.random.default_rng(4).integers(6, 97, size=20)]
    _tiny_run(tiny_dir, monkeypatch, env, prompt, 20, 3)
    header, recs = _read_trace(trace_path)
    layers = sorted(m["layer"] for m in header["layers"])
    assert all(not m["resident"] for m in header["layers"])
    per_fwd = {}
    for f, _, _ in recs:
        per_fwd.setdefault(f[1], []).append(f[3])
    assert all(sorted(v) == layers for v in per_fwd.values()), "one record per streamed layer per forward"
