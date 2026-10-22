"""TF_KV_PAGED=1: prompt-cache entries in KV pages (``families/qwen4_exp/paged_kv.py``) against plain AttentionCache
copies of the same random KV, on the CPU; and the plain AttentionCache without the flag."""

from __future__ import annotations

import contextlib
import gc
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.qwen4_exp import model as q4  # noqa: E402
from tensorfold.families.qwen4_exp import paged_kv as P  # noqa: E402

KVH, D, DI = 2, 256, 128
PAGED = {"TF_KV_PAGED": "1", "TF_CACHE_COMPACT": "1"}
ROOMY = {"TF_KV_POOL_PAGES": "4096", **PAGED}       # more pages than these tests use: no eviction
NAMES = ("TF_KV_PAGED", "TF_CACHE_COMPACT", "TF_KV_POOL_PAGES", "TF_CKPT_SHARE")


@contextlib.contextmanager
def env(**values: str):
    """These settings (the others unset), the CPU and fresh pools for one test."""

    saved = {name: os.environ.get(name) for name in NAMES}
    device = mx.default_device()
    for name in NAMES:
        os.environ.pop(name, None)
    os.environ.update(values)
    mx.set_default_device(mx.cpu)
    P.reset()
    try:
        yield
    finally:
        gc.collect()
        P.reset()
        mx.set_default_device(device)
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def chunk(rng: np.random.Generator, n: int) -> tuple[mx.array, mx.array, mx.array]:
    def f(*shape: int) -> mx.array:
        return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)

    return f(1, KVH, n, D), f(1, KVH, n, D), f(1, n, DI)


def same(a: mx.array, b: mx.array) -> bool:
    return a.shape == b.shape and bool(mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())


def step(rng: np.random.Generator, live: q4.AttentionCache, ref: q4.AttentionCache, n: int) -> None:
    """The same ``n`` new positions into both caches."""

    k, v, ik = chunk(rng, n)
    live.update(k, v, ik)
    ref.update(k, v, ik)
    if live.offset > 2048:          # stands in for the indexer's pooled blocks (entries keep raw keys past them)
        live.pooled = mx.zeros((1, live.offset // 4, DI), mx.bfloat16)
        ref.pooled = live.pooled
    mx.eval(live.keys, live.values, live.index_keys, ref.keys, ref.values, ref.index_keys)


def snap(ref: q4.AttentionCache) -> tuple[mx.array, mx.array, mx.array]:
    n = ref.offset
    out = (mx.contiguous(ref.keys[:, :, :n]), mx.contiguous(ref.values[:, :, :n]),
           mx.contiguous(ref.index_keys[:, :n]))
    mx.eval(*out)
    return out


def live_matches(live: P.HybridAttentionCache, ref: q4.AttentionCache) -> bool:
    n, lo = ref.offset, live.index_from         # raw index keys below index_from are zeros (never read)
    return (live.offset == n and same(live.keys[:, :, :n], ref.keys[:, :, :n])
            and same(live.values[:, :, :n], ref.values[:, :, :n])
            and same(live.index_keys[:, lo:n], ref.index_keys[:, lo:n]))


def entry_matches(entry: P.PagedAttentionCache, keys: mx.array, values: mx.array, index_keys: mx.array) -> bool:
    n, lo = entry.offset, entry.index_from
    plain = entry.snapshot_copy()
    return (type(entry) is P.PagedAttentionCache and same(plain.keys, keys[:, :, :n])
            and same(plain.values, values[:, :, :n]) and same(plain.index_keys[:, lo:n], index_keys[:, lo:n]))


def raises(error: type[BaseException], call) -> bool:
    try:
        call()
    except error:
        return True
    return False


def test_unset_is_plain_attention_cache() -> None:
    layers = [SimpleNamespace(is_linear=i % 4 != 3) for i in range(8)]
    want = [q4.LinearCache if layer.is_linear else q4.AttentionCache for layer in layers]
    with env():
        assert [type(c) for c in q4.Qwen4Exp.make_cache(SimpleNamespace(layers=layers))] == want
        assert type(q4.attention_cache(3)) is q4.AttentionCache
    with env(TF_KV_PAGED="0", TF_CACHE_COMPACT="1"):
        assert [type(c) for c in q4.Qwen4Exp.make_cache(SimpleNamespace(layers=layers))] == want


def test_paged_needs_compact_copies_and_replaces_ckpt_share() -> None:
    from tensorfold.engine import lane_family

    with env(TF_KV_PAGED="1"):
        assert raises(RuntimeError, lambda: q4.attention_cache(3))
    with env(TF_CKPT_SHARE="1", **PAGED):
        assert type(q4.attention_cache(3)) is P.HybridAttentionCache
        assert not lane_family._ckpt_share()
    with env(TF_CKPT_SHARE="1", TF_CACHE_COMPACT="1"):
        assert lane_family._ckpt_share()


def test_conversation_through_the_store() -> None:
    """A prompt with three prefill checkpoints, decode rows, the stored copy, a resume, more prompt rows, drafting
    rollbacks, a checkpoint, a rollback into the source pages and a rewrite, another checkpoint. The live K/V equal
    the plain cache's throughout, each entry equals the plain copy taken with it and stays so, a checkpoint takes
    pages only for the positions past the previous one, and every page is free once the caches are gone."""

    with env(**ROOMY):
        rng = np.random.default_rng(11)
        live, ref = q4.attention_cache(0), q4.AttentionCache()
        assert type(live) is P.HybridAttentionCache
        pool = live.pool
        live.reserve(30100)
        ref.reserve(30100)
        entries = []
        done = prev = 0
        for cut in (10240, 20480, 30000):
            while done < cut:
                m = min(2048, cut - done)
                step(rng, live, ref, m)
                done += m
            before = pool.used()
            entry, at = live.store_copy(), snap(ref)
            assert entry_matches(entry, *at)
            entries.append((entry, at))
            del entry
            if prev:
                assert pool.used() - before <= -(-(cut - prev) // P.PAGE) + 1
            prev = cut
        assert live_matches(live, ref)
        for _ in range(37):
            step(rng, live, ref, 1)
        final, final_ref = live.store_copy(), snap(ref)
        assert entry_matches(final, *final_ref)
        assert final.index_from > 0
        del live, ref
        gc.collect()
        live = final.copy_compact(final.offset + 4096)       # the next request resumes from the stored entry
        assert type(live) is P.HybridAttentionCache and live.keys.shape[2] >= final.offset + 4096
        ref = q4.AttentionCache()
        ref.update(*final_ref)
        ref.reserve(final.offset + 4096)
        mx.eval(ref.keys, ref.values, ref.index_keys)
        assert live_matches(live, ref)
        step(rng, live, ref, 2048)
        step(rng, live, ref, 952)
        for _ in range(10):
            step(rng, live, ref, 4)
            live.trim(2)
            ref.trim(2)
        assert live_matches(live, ref)
        second = live.store_copy()
        assert entry_matches(second, *snap(ref))
        back = live.offset - (final.offset - 100)
        live.trim(back)
        ref.trim(back)
        step(rng, live, ref, 300)
        assert live_matches(live, ref)
        third = live.store_copy()
        assert entry_matches(third, *snap(ref))
        assert entry_matches(final, *final_ref)
        assert entry_matches(entries[1][0], *entries[1][1])
        del second, third, final, entries, live, ref
        gc.collect()
        assert pool.used() == 0


def test_stored_copy_writes_only_new_positions() -> None:
    """A finished 60K-position request resumed from a 59K entry: its stored copy takes pages only for the positions
    past the entry's full pages."""

    with env(**ROOMY):
        rng = np.random.default_rng(12)
        live, ref = q4.attention_cache(0), q4.AttentionCache()
        for _ in range(29):
            step(rng, live, ref, 2048)
        base = live.store_copy()
        del live
        gc.collect()
        live = base.copy_compact(base.offset + 4096)
        for _ in range(8):
            step(rng, live, ref, 128)
        before = live.pool.used()
        entry = live.store_copy()
        assert live.pool.used() - before <= 6
        assert entry_matches(entry, *snap(ref))


def test_prompt_cache_store_evicts_for_pages() -> None:
    """``CheckpointStore.insert`` stores a live cache as pages (``store_copy``); ``unique_cache_bytes`` counts a page
    shared by two entries once; a full pool evicts entries, the one last used too; with nothing left to evict the
    stored copy is contiguous, and a request resumed from it is paged again (``adopt``)."""

    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.server.app import CheckpointStore, unique_cache_bytes

    with env(TF_KV_POOL_PAGES="64", **PAGED):
        rng = np.random.default_rng(13)
        store = CheckpointStore(8, copier=LaneEngine.copy_single_cache)
        P.attach(store.evict_lru, 0)
        live, ref = q4.attention_cache(0), q4.AttentionCache()
        for _ in range(5):
            step(rng, live, ref, 2048)                          # 10,240 positions: 40 pages
        store.insert(list(range(10240)), [live], last_prompt=[])
        del live
        first = store._entries[0].cache[0]
        assert type(first) is P.PagedAttentionCache and len(first.pages) == 40
        pool = first.pool
        cached, (resumed,), _ = store.match(list(range(12500)), reserve=4096)
        assert cached == 10240 and type(resumed) is P.HybridAttentionCache and len(resumed.src_pages) == 40
        step(rng, resumed, ref, 2048)
        store.insert(list(range(12288)), [resumed], last_prompt=[])
        del resumed
        gc.collect()
        second = store._entries[0].cache[0]
        assert second.pages[:40] == first.pages and pool.used() == 48
        arrays = sum(a.nbytes for e in (first, second) for a in e.state)
        assert unique_cache_bytes([[first], [second]]) == 48 * pool.page_bytes() + arrays
        del first, second
        # 30 pages more with 16 free: both entries go (the older one frees no page: its pages are the newer one's)
        other, scratch = q4.attention_cache(0), q4.AttentionCache()
        for _ in range(4):
            step(rng, other, scratch, 1920)
        evictions = store.evictions
        store.insert(list(range(1, 7681)), [other], last_prompt=[])
        del other
        gc.collect()
        assert store.evictions - evictions == 2 and len(store) == 1 and pool.used() == 30
        # 70 pages do not fit a 64-page pool: the copy is contiguous at once, and the paged entry is not evicted for it
        big, bigref = q4.attention_cache(0), q4.AttentionCache()
        for _ in range(10):
            step(rng, big, bigref, 1792)
        evictions = store.evictions
        store.insert(list(range(2, 17922)), [big], last_prompt=[])
        del big
        gc.collect()
        plain = store._entries[0].cache[0]
        assert type(plain) is q4.AttentionCache and len(store) == 2 and store.evictions == evictions
        assert pool.used() == 30
        assert same(plain.keys[:, :, :plain.offset], bigref.keys[:, :, :bigref.offset])
        # a request resumed from a contiguous entry is paged again from its next stored copy
        small, smallref = q4.AttentionCache(), q4.AttentionCache()
        step(rng, small, smallref, 3000)
        state, adopted = P.adopt([q4.LinearCache(), small])
        assert type(state) is q4.LinearCache and type(adopted) is P.HybridAttentionCache and adopted.layer == 1
        assert adopted.keys is small.keys
        assert entry_matches(adopted.store_copy(), *snap(smallref))


def test_prompt_cache_keeps_entries_too_big_for_the_pool() -> None:
    """A pool smaller than one prompt: each stored copy is contiguous and no entry is evicted for pages, so the prompt
    cache keeps both entries and a later request resumes from one."""

    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.server.app import CheckpointStore

    with env(TF_KV_POOL_PAGES="32", **PAGED):
        rng = np.random.default_rng(16)
        store = CheckpointStore(8, copier=LaneEngine.copy_single_cache)
        P.attach(store.evict_lru, 0)
        refs = {}
        for first in (0, 1):
            live, ref = q4.attention_cache(0), q4.AttentionCache()
            refs[first] = ref
            for _ in range(5):
                step(rng, live, ref, 2048)                      # 10,240 positions: 40 pages, more than the pool
            store.insert(list(range(first, first + 10240)), [live], last_prompt=[])
            del live
            gc.collect()
        assert len(store) == 2 and store.evictions == 0
        assert all(type(e.cache[0]) is q4.AttentionCache for e in store._entries)
        assert P.pool(0).used() == 0
        hit = store.match(list(range(10300)))
        assert hit is not None and hit[0] == 10240 and hit[1][0].offset == 10240
        assert same(hit[1][0].keys[:, :, :10240], refs[0].keys[:, :, :10240])
        assert same(hit[1][0].values[:, :, :10240], refs[0].values[:, :, :10240])


def test_full_pool_gives_back_a_partial_copy() -> None:
    """A copy that fits the pool but finds it full of a pinned entry's pages: evict_lru has nothing to evict, the pages
    the copy took are given back, the copy is contiguous, and the pinned entry keeps its pages."""

    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.server.app import CheckpointStore

    with env(TF_KV_POOL_PAGES="64", **PAGED):
        rng = np.random.default_rng(17)
        store = CheckpointStore(8, copier=LaneEngine.copy_single_cache)
        P.attach(store.evict_lru, 0)
        live, ref = q4.attention_cache(0), q4.AttentionCache()
        for _ in range(5):
            step(rng, live, ref, 2048)                          # 10,240 positions: 40 pages
        store.insert(list(range(10240)), [live], last_prompt=[], pinned=True)
        del live
        gc.collect()
        pinned = store._entries[0].cache[0]
        assert type(pinned) is P.PagedAttentionCache and len(pinned.pages) == 40
        pool = pinned.pool
        assert pool.used() == 40
        other, scratch = q4.attention_cache(0), q4.AttentionCache()
        for _ in range(4):
            step(rng, other, scratch, 1920)                     # 30 pages: fits 64, but only 24 are free
        evictions = store.evictions
        store.insert(list(range(1, 7681)), [other], last_prompt=[])
        del other
        gc.collect()
        plain = store._entries[0].cache[0]
        assert type(plain) is q4.AttentionCache and plain.offset == 7680
        assert len(store) == 2 and store.evictions == evictions and pool.used() == 40
        assert store._entries[1].cache[0] is pinned and pinned.pages and len(pinned.pages) == 40
        assert same(plain.keys[:, :, :7680], scratch.keys[:, :, :7680])


def test_disk_snapshot_of_a_paged_entry() -> None:
    """``save_snapshot`` writes a paged entry as a plain AttentionCache (``snapshot_copy``) that reads back equal."""

    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot

    with env(**PAGED), tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(14)
        live, ref = q4.attention_cache(0), q4.AttentionCache()
        for _ in range(4):
            step(rng, live, ref, 2048)
        step(rng, live, ref, 100)
        entry = live.store_copy()
        n, lo = entry.offset, entry.index_from
        assert lo > 0
        path = save_snapshot(Path(tmp), "model", list(range(n)), [entry])
        tokens, (loaded,) = load_snapshot(path, "model")
        assert tokens == list(range(n)) and type(loaded) is q4.AttentionCache and loaded.offset == n
        assert same(loaded.keys, ref.keys[:, :, :n]) and same(loaded.values, ref.values[:, :, :n])
        assert same(loaded.index_keys[:, lo:n], ref.index_keys[:, lo:n]) and same(loaded.pooled, ref.pooled)


def test_disk_snapshot_of_a_live_cache() -> None:
    """A live cache's snapshot is a plain AttentionCache (no source-page ids leave the process), keeps ``index_from``,
    and is adopted back with no source pages; a rollback below its kept raw index keys is refused."""

    from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot

    with env(**PAGED), tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(15)
        live, ref = q4.attention_cache(0), q4.AttentionCache()
        for _ in range(4):
            step(rng, live, ref, 2048)
        entry = live.store_copy()
        resumed = entry.copy_compact(512)
        assert resumed.src_pages and resumed.index_from > 0
        n = resumed.offset
        path = save_snapshot(Path(tmp), "model", list(range(n)), [resumed])
        tokens, (loaded,) = load_snapshot(path, "model")
        assert type(loaded) is q4.AttentionCache and not hasattr(loaded, "src_pages")
        assert loaded.index_from == resumed.index_from and same(loaded.keys[:, :, :n], ref.keys[:, :, :n])
        (adopted,) = P.adopt([loaded])
        assert type(adopted) is P.HybridAttentionCache and adopted.src_pages == []
        with pytest.raises(RuntimeError):
            adopted.trim(adopted.offset - adopted.index_from + 8)


def test_checkpoint_copy_takes_the_stored_form() -> None:
    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.engine.lane_family import checkpoint_copy

    engine = SimpleNamespace(copy_single_cache=LaneEngine.copy_single_cache)
    with env(**PAGED):
        rng = np.random.default_rng(15)
        live, ref = q4.attention_cache(3), q4.AttentionCache()
        step(rng, live, ref, 3000)
        state = q4.LinearCache()
        state.conv = mx.ones((1, 3, 8))
        out = checkpoint_copy(engine, [state, live])
        assert type(out[0]) is q4.LinearCache and out[0] is not state and same(out[0].conv, state.conv)
        assert type(out[1]) is P.PagedAttentionCache and entry_matches(out[1], *snap(ref))
    with env():
        plain, ref = q4.AttentionCache(), q4.AttentionCache()
        step(np.random.default_rng(16), plain, ref, 300)
        (copy,) = checkpoint_copy(engine, [plain])
        assert type(copy) is q4.AttentionCache and same(copy.keys[:, :, :300], ref.keys[:, :, :300])
