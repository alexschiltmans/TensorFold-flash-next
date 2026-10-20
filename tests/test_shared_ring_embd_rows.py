"""TF_EXPERT_RING=shared and TF_TRUNK_EMBD_ROWS=1, each against the toggle off.

Uses test_expert_stream's synthetic GGUF (four layers at the real expert dims, every cut240 codec pairing, random
blocks: every expert and projection has its own bytes) and its tiny model, plus a tiny trunk GGUF holding only
token_embd (IQ4_XS) with every other trunk entry kept from the checkpoint (``fallback``). Outputs are compared bit
for bit. GPU kernels, small arrays.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
kq = pytest.importorskip("mlx_kquant")

from tensorfold.families.qwen4_exp import expert_stream as es  # noqa: E402
from tests.test_expert_stream import (  # noqa: E402,F401 - fixtures
    E, LAYERS, SLOTS, TINY_LAYERS, TOP, TYPE_ID, FakeMoE, _small, bits, cfg, experts_of, expert_tensors, gguf,
    make_switch, route, stock, write_gguf, x_of)


def _stream(gx, layer, **kw):
    return es.ExpertStream(gx, cfg(ring="shared", **kw), layers=[layer])


def _stale(stream, seed=1):
    for buf in stream.ring.bufs.values():
        buf.host[:] = np.random.default_rng(seed).integers(0, 2**15, size=buf.host.size).astype(buf.host.dtype)


def _file_slice(path: Path, off: int, n: int) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(off)
        return np.frombuffer(f.read(n), dtype=np.uint8)


def _arena_holds_the_file(stream, path: Path) -> int:
    """Every resident kquant slot holds its expert's file bytes; returns the slots checked."""

    arena, n = stream.arenas[0], 0
    for e in arena.resident().tolist():
        s = int(arena.slot_of[e])
        for proj, plan in arena.plans.items():
            if plan.codec is None:
                continue
            got = np.asarray(arena.bufs[proj]["weight"].slot(s, plan.file_bytes))
            assert np.array_equal(got, _file_slice(path, plan.offset + e * plan.file_bytes, plan.file_bytes)), (e, proj)
            n += 1
    return n


# -- the shared ring ------------------------------------------------------------------
def test_settings_ring_mode_and_sizes(gguf):
    assert es.settings({"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16"})["ring"] == "split"
    assert es.settings({"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16", "TF_EXPERT_RING": "shared"})["ring"] \
        == "shared"
    with pytest.raises(ValueError):
        es.settings({"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16", "TF_EXPERT_RING": "one"})
    gx = experts_of(gguf)
    split = es.ExpertStream(gx, cfg())
    shared = es.ExpertStream(gx, cfg(ring="shared"))
    assert "MiB split" in split.summary() and "MiB shared" in shared.summary()
    plans = es.plans_from(gx)
    largest = max(p.experts * p.served_bytes for row in plans for p in row.values())
    pair = shared.ring.bufs[("pair", "weight")].nbytes
    assert shared.ring.nbytes - pair <= 2 * largest < split.ring.nbytes             # two codec families at most
    print(f"\nring split {split.ring.nbytes / 2**20:.1f} MiB, shared {shared.ring.nbytes / 2**20:.1f} MiB "
          f"(pair {pair / 2**20:.1f} MiB)")
    split.close()
    shared.close()


@pytest.mark.parametrize("layer", range(len(LAYERS)))
def test_shared_ring_chunks_are_exact_over_stale_bytes_and_warm_per_projection(gguf, stock, layer):
    gx = experts_of(gguf)
    ref = make_switch(gx, layer, stock, E)
    stream = _stream(gx, layer)
    _stale(stream)
    moe = FakeMoE(make_switch(gx, layer, None, SLOTS))
    hook = stream.layer(layer)
    rng = np.random.default_rng(400 + layer)
    for _ in range(1):                                                  # a partly warm arena (10 of 16)
        x, idx = x_of(rng, 1), route(rng, 1)
        hook.done(hook(moe, x, mx.array(idx)))
    assert 0 < stream.arenas[0].resident().size < SLOTS
    held = stream.arenas[0].slot_of.copy()
    same = np.tile(rng.choice(E, TOP, replace=False), (300, 1)).astype(np.uint32)[None]   # one route, every row
    for rows, final, idx in ((2048, False, None), (300, False, same), (600, None, None)):
        stream.final_chunk = final
        x = x_of(rng, rows)
        idx = route(rng, rows, rng.choice(E, 20, replace=False)) if idx is None else idx
        got = hook(moe, x, mx.array(idx))
        hook.done(got)
        assert np.array_equal(bits(ref(x, mx.array(idx))), bits(got)), rows
        rec = stream.calls[-1]
        assert rec["path"] == "ring" and not rec["compact"]
        if final is False:
            assert "warmed" not in rec and np.array_equal(stream.arenas[0].slot_of, held)
        else:
            assert rec["warmed"] > 0                                    # evictions of hits staged later included
    assert _arena_holds_the_file(stream, gguf) > 0
    resident = stream.arenas[0].resident()                              # warmed slots: exact hits, no reads
    idx = np.array(rng.choice(resident, TOP, replace=False), dtype=np.uint32)[None, None]
    x = x_of(rng, 1)
    got = hook(moe, x, mx.array(idx))
    hook.done(got)
    assert np.array_equal(bits(ref(x, mx.array(idx))), bits(got)) and stream.calls[-1]["misses"] == 0
    stream.close()


@pytest.mark.parametrize("layer,glu", [(i, "1") for i in range(len(LAYERS))] + [(0, "0"), (1, "0")])
def test_shared_ring_short_calls_beyond_the_slots_run_the_fused_glu_on_compact_stacks(gguf, stock, layer, glu,
                                                                                     monkeypatch):
    monkeypatch.setenv("TF_FLASH_KQ_GLU", glu)                         # 0: SwitchGLU's unsorted path, staged
    gx = experts_of(gguf)
    ref = make_switch(gx, layer, stock, E)
    stream = _stream(gx, layer, prefill_rows=4)                         # rows < 4 count as a final chunk (warm)
    _stale(stream, 2)
    moe = FakeMoE(make_switch(gx, layer, None, SLOTS))
    hook = stream.layer(layer)
    rng = np.random.default_rng(500 + layer)
    paths = []
    for rows in (2, 6, 3, 1, 5, 2, 7, 1):
        x, idx = x_of(rng, rows), route(rng, rows)
        got = hook(moe, x, mx.array(idx))
        hook.done(got)
        assert np.array_equal(bits(ref(x, mx.array(idx))), bits(got)), rows
        rec = stream.calls[-1]
        paths.append((rec["path"], rec.get("compact")))
    assert (("ring", True) in paths) == (glu == "1") and ("arena", None) in paths and ("ring", False) in paths
    assert sum(c.get("warmed", 0) for c in stream.calls) > 0
    assert _arena_holds_the_file(stream, gguf) > 0
    stream.close()


@pytest.mark.parametrize("fence", [True, False])
def test_a_stage_waits_for_the_previous_projections_eval(gguf, stock, fence):
    gx = experts_of(gguf)
    ref = make_switch(gx, 0, stock, E)
    stream = _stream(gx, 0)
    stream.fence = fence
    moe = FakeMoE(make_switch(gx, 0, None, SLOTS))
    hook = stream.layer(0)
    rng = np.random.default_rng(6)
    x, idx = x_of(rng, 256), route(rng, 256)
    got = hook(moe, x, mx.array(idx))                                   # without the evals up reads gate's bytes
    hook.done(got)
    assert np.array_equal(bits(ref(x, mx.array(idx))), bits(got)) == fence
    stream.close()


def test_shared_ring_pinned(gguf, stock):
    gx = experts_of(gguf)
    stream = _stream(gx, 1, pin=True)
    assert stream.pin is not None and stream.pin.nbytes >= stream.ring.nbytes
    ref = make_switch(gx, 1, stock, E)
    moe = FakeMoE(make_switch(gx, 1, None, SLOTS))
    hook = stream.layer(1)
    rng = np.random.default_rng(8)
    x, idx = x_of(rng, 100), route(rng, 100)
    got = hook(moe, x, mx.array(idx))
    hook.done(got)
    assert np.array_equal(bits(ref(x, mx.array(idx))), bits(got))
    stream.close()


# -- a tiny model with a trunk GGUF -----------------------------------------------------
VOCAB = 4096


def _embd_blocks(rng, rows: int, dims: int) -> bytes:
    raw = rng.integers(0, 256, size=(rows * dims // 256, 136), dtype=np.uint8)          # IQ4_XS: 256 a block
    raw[:, :2] = rng.uniform(0.0001, 0.0004, size=raw.shape[0]).astype(np.float16).view(np.uint8).reshape(-1, 2)
    return raw.tobytes()


@pytest.fixture(scope="module")
def trunk_dir(tmp_path_factory) -> Path:
    from mlx.utils import tree_flatten

    from tensorfold.families.qwen4_exp import model as q4
    from tests.test_qwen4_exp_family import TEXT

    text = dict(TEXT, hidden_size=256, moe_intermediate_size=256, num_experts=32, num_experts_per_tok=4,
                shared_expert_intermediate_size=64, linear_key_head_dim=32, linear_value_head_dim=32, vocab_size=VOCAB)
    d = tmp_path_factory.mktemp("trunk")
    mx.random.seed(21)
    model = q4.Qwen4Exp(q4.Config.from_dict({"text_config": text}))
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        if ".switch_mlp." in name:
            continue
        if name.endswith("norm.weight") or "layernorm" in name or "hc_norm" in name or name.endswith("norm_key.weight"):
            value = 0.1 * mx.random.normal(value.shape)
        weights[f"language_model.{name}"] = value.astype(mx.bfloat16)
    mx.save_safetensors(str(d / "model.safetensors"), weights)
    rng = np.random.default_rng(22)
    keep = {str(i): sorted(rng.choice(32, E, replace=False).tolist()) for i in range(len(TINY_LAYERS))}
    (d / "cut.json").write_text(json.dumps({"keep_local": keep}))
    write_gguf(d / "experts.gguf", expert_tensors(rng, TINY_LAYERS, 256, 256, E))
    write_gguf(d / "trunk.gguf", [("token_embd.weight", (256, VOCAB), TYPE_ID["IQ4_XS"], _embd_blocks(rng, VOCAB, 256))])
    (d / "config.json").write_text(json.dumps({
        "text_config": text, "eos_token_id": 5,
        "experts_source": {"format": "gguf-kquant", "file": "experts.gguf"}, "experts_cut": {"file": "cut.json"},
        "trunk_source": {"format": "gguf-kquant", "file": str(d / "trunk.gguf"), "require_zero_copy": True,
                         "fallback": [r"^(?!model\.embed_tokens\.weight$)"]}}))
    return d


_KEYS = ("TF_EXPERT_", "TF_TRUNK_", "TF_FLASH_INFLIGHT")


def _load(trunk_dir: Path, monkeypatch, env: dict[str, str]):
    import mlx_lm.utils

    from tensorfold.families.qwen4_exp import model as q4

    for key in [k for k in os.environ if k.startswith(_KEYS)]:
        monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", lambda *a, **k: None)
    return q4.load(trunk_dir)[0]


def _run(model, prompt: list[int], chunk: int, steps: int):
    cache = model.make_cache()
    stream = model.__dict__.get("expert_stream")
    for begin in range(0, len(prompt), chunk):
        if stream is not None:
            stream.final_chunk = begin + chunk >= len(prompt)
        out = model(np.array([prompt[begin:begin + chunk]]), cache)
        mx.eval(out)
    got, tokens = [], []
    for _ in range(steps):
        last = out[0, -1]
        got.append(bits(last))
        tokens.append(int(mx.argmax(last).item()))
        out = model(np.array([[tokens[-1]]]), cache)
    return got, tokens


STREAM = {"TF_EXPERT_STREAM": "1", "TF_EXPERT_ARENA_SLOTS": "16", "TF_EXPERT_DEBUG": "1", "TF_EXPERT_PREFILL_ROWS": "32",
          "TF_FLASH_INFLIGHT": "1", "TF_FLASH_INFLIGHT_PARTS": "1"}


@pytest.mark.parametrize("prompt_len, chunk", [(23, 23), (80, 32)])
def test_tiny_model_toggles_on_equal_toggles_off_bit_for_bit(trunk_dir, monkeypatch, capsys, prompt_len, chunk):
    rng = np.random.default_rng(prompt_len)
    prompt = [int(t) for t in rng.integers(6, 97, size=prompt_len)]
    arms = {
        "stock": {},
        "embd": {"TF_TRUNK_EMBD_ROWS": "1"},
        "stream": dict(STREAM),
        "stream-embd": dict(STREAM, TF_TRUNK_EMBD_ROWS="1"),
        "stream-ring": dict(STREAM, TF_EXPERT_RING="shared"),
        "stream-both": dict(STREAM, TF_EXPERT_RING="shared", TF_TRUNK_EMBD_ROWS="1"),
    }
    out: dict[str, tuple] = {}
    for name, env in arms.items():
        model = _load(trunk_dir, monkeypatch, env)
        from mlx_kquant.nn import KQuantEmbedding

        from tensorfold.families.qwen4_exp.gguf_embd import GGUFEmbedRows

        emb = model.model.embed_tokens
        assert isinstance(emb, GGUFEmbedRows if "TF_TRUNK_EMBD_ROWS" in env else KQuantEmbedding), name
        out[name] = _run(model, prompt, chunk, 24)
        stream = model.__dict__.get("expert_stream")
        if stream is not None:
            assert stream.ring.shared == ("TF_EXPERT_RING" in env)
            assert stream.totals["calls_ring"] > 0
            stream.close()
        del model
    log = capsys.readouterr().out
    assert "token_embd as CPU row reads" in log and "MiB shared" in log
    base_bits, base_tokens = out["stock"]
    assert len(set(base_tokens)) > 3
    for name, (got_bits, got_tokens) in out.items():
        assert got_tokens == base_tokens, name
        for step, (a, b) in enumerate(zip(base_bits, got_bits)):
            assert np.array_equal(a, b), (name, step)

def test_embd_rows_frees_the_table_from_mlx_and_is_fast(trunk_dir, monkeypatch, capsys):
    import mlx_kquant as kq

    def loaded(env):
        gc.collect()
        mx.clear_cache()
        views = kq.zero_copy_view_count()
        before = mx.get_active_memory()
        model = _load(trunk_dir, monkeypatch, env)
        return model, mx.get_active_memory() - before, kq.zero_copy_view_count() - views

    base, base_mem, base_views = loaded({})
    table = VOCAB * 136
    del base
    rows, rows_mem, rows_views = loaded({"TF_TRUNK_EMBD_ROWS": "1"})
    emb = rows.model.embed_tokens
    assert emb.nbytes == table and base_views - rows_views == 1                # token_embd only
    assert rows.__dict__["gguf_trunk"].report["cpu_rows"] == ["model.embed_tokens.weight"]
    drop = base_mem - rows_mem
    assert drop >= 0.9 * table, (base_mem, rows_mem)                # page rounding aside
    times = {}
    for n in (1, 2048):
        ids = mx.array(np.random.default_rng(n).integers(0, VOCAB, size=(1, n)).astype(np.int32))
        mx.eval(emb(ids))
        t0 = time.perf_counter()
        for _ in range(20):
            mx.eval(emb(ids))
        times[n] = (time.perf_counter() - t0) / 20 * 1e3
        np_rows = time.perf_counter()
        for _ in range(20):
            emb.data[np.asarray(ids).reshape(-1)]
        times[f"{n}-gather"] = (time.perf_counter() - np_rows) / 20 * 1e3
    with pytest.raises(RuntimeError):
        emb.as_linear(mx.zeros((1, 256), dtype=mx.bfloat16))
    print(f"\nMLX active after load: {base_mem} -> {rows_mem} bytes (drop {drop}, table {table}); call ms {times}")


def test_embd_rows_refusals(tiny_dir_free, trunk_dir, monkeypatch):
    for env in ({"TF_TRUNK_EMBD_ROWS": "2"}, {"TF_TRUNK_EMBD_ROWS": "1", "TF_FLASH_REF_DRAFT": "pad"},
                {"TF_TRUNK_EMBD_ROWS": "1", "TF_FLASH_MTP_GGUF": "/x.gguf"}):
        with pytest.raises(ValueError):
            _load(trunk_dir, monkeypatch, env)
        for key in env:
            monkeypatch.delenv(key)
    with pytest.raises(ValueError, match="trunk_source"):
        _load(tiny_dir_free, monkeypatch, {"TF_TRUNK_EMBD_ROWS": "1"})


@pytest.fixture
def tiny_dir_free(trunk_dir, tmp_path) -> Path:
    """The trunk model's folder without trunk_source."""

    d = tmp_path / "free"
    d.mkdir()
    for f in trunk_dir.iterdir():
        if f.name != "config.json":
            (d / f.name).symlink_to(f)
    config = json.loads((trunk_dir / "config.json").read_text())
    config.pop("trunk_source")
    (d / "config.json").write_text(json.dumps(config))
    return d
