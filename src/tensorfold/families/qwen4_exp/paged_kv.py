"""Prompt-cache KV in pages for Flash Next's sparse-attention layers (TF_KV_PAGED=1, with TF_CACHE_COMPACT=1).

Without the flag each prompt-cache entry and each prefill checkpoint holds its own contiguous copy of every
sparse-attention layer's keys and values, so a conversation's prefix is held once for every entry that contains it.
With it, each sparse-attention layer has a pool of 256-position pages (``KVPool``), and a stored copy
(``PagedAttentionCache``) is a page table into that pool. The request being served keeps contiguous buffers
(``HybridAttentionCache`` is an ``AttentionCache``: decode and prefill run on the same buffers as without the flag)
and references the full pages that hold its first positions (``src_pages``: the entry it was resumed from, or its
last stored copy). Its next stored copy references those pages and writes only the positions after them, so a
prefill's checkpoints, the entry a request resumes from and the entry its finished cache becomes share their full
pages. A request resumed from an entry gets contiguous buffers filled from the entry's pages.

Pages are reference counted, and a cache gives its references back in ``__del__``: CPython runs it as soon as the
last reference to the cache goes, so an entry dropped from the prompt cache returns its pages at once. A pool
allocates its buffers 32 pages at a time, up to its size. When no page is free it asks the server's prompt cache to
evict its least recently used entry (``attach``); a copy that still finds no page is made contiguous instead.
Disk snapshots (``--snapshot-dir``) are written as plain AttentionCache layers (``snapshot_copy``); a request resumed
from one, or from a contiguous copy, is paged again from its next stored copy (``adopt``).

MLX writes a buffer in place only while no other array references it, so only the pool holds its buffers: pages are
written from evaluated temporaries and read back as copies, never as views. An entry keeps the indexer's raw keys
only from INDEX_KEEP positions below its last pooled block (``_index_from``); a cache resumed from it holds zeros
below that.

    TF_KV_PAGED=1          prompt-cache entries and prefill checkpoints in pages (needs TF_CACHE_COMPACT=1)
    TF_KV_POOL_PAGES=N     pages a layer's pool holds at most (default: the served context window in pages, plus 32)
"""

from __future__ import annotations

import os
import threading
from typing import Any

import mlx.core as mx

from tensorfold.families.qwen4_exp.model import AttentionCache

PAGE = 256              # positions a page
CHUNK = 32              # pages a pool buffer: buffers are allocated as their pages are first needed
SPARE_PAGES = 32        # the default pool's pages beyond one window: partial last pages, a prefill's checkpoints
DEFAULT_CONTEXT = 32768  # the window pools are sized for until the server names its own (or when it has none)
INDEX_KEEP = 1024       # raw index-key positions an entry keeps below its last pooled block
RATIO = 4               # positions an indexer block pools (indexer_compress_ratio)


def enabled() -> bool:
    return os.environ.get("TF_KV_PAGED", "").strip() not in ("", "0")


class PoolFull(RuntimeError):
    """No free page in a layer's pool, and the prompt cache had no entry left to evict."""


def _pool_pages(context: int) -> int:
    """TF_KV_POOL_PAGES, or ``context`` positions in pages plus SPARE_PAGES; rounded up to whole buffers."""

    pages = int(os.environ.get("TF_KV_POOL_PAGES", "0") or 0)
    if pages <= 0:
        pages = -(-int(context or DEFAULT_CONTEXT) // PAGE) + SPARE_PAGES
    return -(-pages // CHUNK) * CHUNK


class KVPool:
    """One sparse-attention layer's pages: K and V buffers of CHUNK pages each, a reference count per page, a free
    list. Page ``p`` is positions [(p % CHUNK) * PAGE, + PAGE) of buffer ``p // CHUNK``."""

    def __init__(self, layer: int, pages: int) -> None:
        self.layer = layer
        self.pages = pages                                  # the most pages this pool allocates
        self.ks: list[mx.array] = []
        self.vs: list[mx.array] = []
        self.refs: list[int] = []
        self.free: list[int] = []
        self.shape: tuple[int, int, Any] | None = None      # (KV heads, head dim, dtype), from the first stored copy
        # reentrant: ``__del__`` gives pages back on whatever thread drops a cache, and a garbage collection that an
        # allocation under the lock triggers may run one; the prompt cache's evict() is called outside it
        self.lock = threading.RLock()

    def _grow(self) -> bool:
        """One more K and V buffer of CHUNK pages, unless the pool has its ``pages`` already (under ``lock``)."""

        if self.shape is None or len(self.refs) + CHUNK > self.pages:
            return False
        heads, dims, dtype = self.shape
        k = mx.zeros((1, heads, CHUNK * PAGE, dims), dtype=dtype)
        v = mx.zeros((1, heads, CHUNK * PAGE, dims), dtype=dtype)
        mx.eval(k, v)
        base = len(self.refs)
        self.ks.append(k)
        self.vs.append(v)
        self.refs.extend([0] * CHUNK)
        self.free.extend(range(base + CHUNK - 1, base - 1, -1))     # pop() hands out the lowest page first
        return True

    def alloc(self) -> int:
        """A free page with one reference: grows the pool, then has the prompt cache evict entries, until one is."""

        while True:
            with self.lock:
                if self.free or self._grow():
                    page = self.free.pop()
                    self.refs[page] = 1
                    return page
            evict = _STATE["evict"]
            if evict is None or not evict():
                raise PoolFull(f"layer {self.layer}: all {self.pages} KV pages are in use (TF_KV_POOL_PAGES)")

    def ref(self, page: int) -> None:
        with self.lock:
            self.refs[page] += 1

    def unref(self, page: int) -> None:
        with self.lock:
            self.refs[page] -= 1
            if self.refs[page] == 0:
                self.free.append(page)

    def used(self) -> int:
        return sum(1 for r in self.refs if r > 0)

    def page_bytes(self) -> int:
        """Bytes of one page's keys and values."""

        if self.shape is None:
            return 0
        heads, dims, dtype = self.shape
        return 2 * heads * PAGE * dims * dtype.size

    def write(self, page: int, keys: mx.array, values: mx.array) -> None:
        """Rows [0, n) of ``page`` = ``keys`` / ``values`` [1, KVH, n, D], evaluated arrays (the write is in place
        while the pool holds the only reference to its buffers). Lazy."""

        c, a = page // CHUNK, (page % CHUNK) * PAGE
        b = a + keys.shape[2]
        with self.lock:
            self.ks[c][:, :, a:b] = keys
            self.vs[c][:, :, a:b] = values

    def gather(self, pages: list[int], n: int) -> tuple[mx.array, mx.array]:
        """Positions [0, n) of ``pages`` (in position order) as K / V [1, KVH, n, D]: new arrays, never views of a
        pool buffer, which would keep it referenced. Lazy."""

        def rows(bufs: list[mx.array]) -> mx.array:
            parts = [bufs[p // CHUNK][:, :, (p % CHUNK) * PAGE:(p % CHUNK + 1) * PAGE] for p in pages]
            if len(parts) == 1:
                return parts[0][:, :, :n] * mx.array(1, parts[0].dtype)
            return mx.concatenate(parts, axis=2)[:, :, :n]

        return rows(self.ks), rows(self.vs)


_POOLS: dict[int, KVPool] = {}
# evict() -> bool: the server's prompt cache drops its least recently used entry that holds pages (False: none does);
# context: the window new pools are sized for
_STATE: dict[str, Any] = {"evict": None, "context": DEFAULT_CONTEXT}


def pool(layer: int) -> KVPool:
    p = _POOLS.get(layer)
    if p is None:
        p = _POOLS[layer] = KVPool(layer, _pool_pages(_STATE["context"]))
    return p


def attach(evict: Any, context_window: int) -> None:
    """The server's prompt cache exists: a pool with no free page calls ``evict()`` until it frees one or returns
    False, and the pools hold ``context_window`` positions in pages plus SPARE_PAGES (0: DEFAULT_CONTEXT), unless
    TF_KV_POOL_PAGES sets their size. Pools made before (the load-time checks') take the new size too. Pools are
    per process and keyed by layer index: one Flash Next model a process."""

    _STATE["evict"] = evict
    _STATE["context"] = int(context_window or 0) or DEFAULT_CONTEXT
    pages = _pool_pages(_STATE["context"])
    for p in _POOLS.values():
        p.pages = max(pages, len(p.refs))
    print(f"[tensorfold] TF_KV_PAGED: prompt-cache KV in pages, up to {pages} pages ({pages * PAGE} positions) "
          f"a sparse-attention layer", flush=True)


def reset() -> None:
    """Forget every pool and the evictor (tests)."""

    _POOLS.clear()
    _STATE.update(evict=None, context=DEFAULT_CONTEXT)


def adopt(cache: list[Any]) -> list[Any]:
    """``cache`` with its plain AttentionCache layers (an entry read from a disk snapshot, or a contiguous copy made
    when a pool was full) as HybridAttentionCache: the same arrays and no source pages, so their next stored copy is
    paged. A layer's position in the list is its layer index (``Qwen4Exp.make_cache``)."""

    out: list[Any] = []
    for layer, item in enumerate(cache):
        if type(item) is AttentionCache:
            live = HybridAttentionCache(layer)
            live.__dict__.update(vars(item))        # keeps an ``index_from`` a paged copy wrote into it
            item = live
        out.append(item)
    return out


def _index_from(pooled: Any, n: int) -> int:
    """First raw index-key position an entry of ``n`` positions keeps. The indexer reads raw keys only to pool the
    blocks not pooled yet, so below its pooled blocks they are read again only after a rollback that drops pooled
    blocks; INDEX_KEEP positions are kept for that. 0 while nothing is pooled (short contexts)."""

    if pooled is None:
        return 0
    return max(0, (min(n, RATIO * int(pooled.shape[1])) - INDEX_KEEP) // RATIO * RATIO)


class PagedAttentionCache(AttentionCache):
    """A stored sparse-attention layer: positions [0, offset) in pages of the layer's pool (``pages``, in position
    order; the last one may be partial), the raw index keys from ``index_from`` on, and the pooled block keys.
    ``keys`` and ``values`` stay None. Never decoded: ``copy_compact`` gives a request contiguous buffers."""

    def __init__(self, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.pages: list[int] = []
        self.index_from = 0

    def __del__(self) -> None:
        # CPython runs this once the last reference goes (the entry evicted or replaced): its pages return to the pool
        pages, self.pages = getattr(self, "pages", []), []
        try:
            for page in pages:
                (self.__dict__.get("_owner") or _POOLS[self.layer]).unref(page)
        except Exception:  # noqa: BLE001 - interpreter shutdown, or no page was ever taken
            pass

    @property
    def pool(self) -> KVPool:
        """The pool this cache's pages belong to, fixed at first use (so ``reset`` cannot hand it another)."""

        owner = self.__dict__.get("_owner")
        if owner is None:
            owner = self.__dict__["_owner"] = pool(self.layer)
        return owner

    def update(self, keys: mx.array, values: mx.array, index_keys: mx.array) -> Any:
        raise RuntimeError("a stored KV-page entry is not decoded: resume from its copy_compact()")

    def page_refs(self) -> list[tuple[int, int]]:
        """(layer, page) of every page this entry references (the server counts a page once across entries)."""

        return [(self.layer, page) for page in self.pages]

    def page_bytes(self) -> int:
        return self.pool.page_bytes()

    @property
    def state(self) -> list[mx.array]:
        out = [] if self.index_keys is None else [self.index_keys]
        return out + ([self.pooled] if self.pooled is not None else [])

    def _contiguous(self, cap: int) -> tuple[mx.array, mx.array, mx.array]:
        """K, V [1, KVH, cap, D] and raw index keys [B, cap, DI] of this entry: positions [0, offset) from the pages
        and the kept index keys (zeros below ``index_from``), zeros from ``offset`` on. Lazy."""

        n = self.offset
        k, v = self.pool.gather(self.pages, n)
        if cap > n:
            pad = (1, k.shape[1], cap - n, k.shape[3])
            k = mx.concatenate([k, mx.zeros(pad, k.dtype)], axis=2)
            v = mx.concatenate([v, mx.zeros(pad, v.dtype)], axis=2)
        lo = self.index_from
        tail = self.index_keys[:, :n - lo]
        batch, _, dims = tail.shape
        parts = (([mx.zeros((batch, lo, dims), tail.dtype)] if lo else []) + [tail]
                 + ([mx.zeros((batch, cap - n, dims), tail.dtype)] if cap > n else []))
        index_keys = mx.concatenate(parts, axis=1) if len(parts) > 1 else tail * mx.array(1, tail.dtype)
        return k, v, index_keys

    def copy_compact(self, reserve: int = 0) -> "HybridAttentionCache":
        """A request resumed from this entry: contiguous buffers with room for ``reserve`` positions (as
        ``AttentionCache.copy_compact``), filled from the pages, referencing the full pages (its first stored copy
        shares them). Evaluated here."""

        live = HybridAttentionCache(self.layer)
        live.offset = n = self.offset
        live.reserved = max(self.reserved, int(reserve))
        live.pooled = None if self.pooled is None else mx.array(self.pooled)
        if n == 0:
            return live
        if self.index_keys is None:
            raise RuntimeError(f"TF_KV_PAGED: a stored entry of {n} positions has no raw index keys")
        cap = ((max(n, int(reserve)) + self.step - 1) // self.step) * self.step if reserve > n else n
        live.keys, live.values, live.index_keys = self._contiguous(cap)
        live.index_from = self.index_from
        mx.eval(live.keys, live.values, live.index_keys)
        live._set_src(self.pages[:n // PAGE], n)
        return live

    def snapshot_copy(self) -> AttentionCache:
        """This entry as a self-contained AttentionCache, for a disk snapshot (``prefix_snapshots.save_snapshot``):
        its pages belong to the pool, so their positions are copied out. Evaluated here."""

        plain = AttentionCache()
        plain.offset, plain.reserved, plain.pooled = self.offset, self.reserved, self.pooled
        plain.index_from = self.index_from      # raw index keys below it are zeros; kept so ``adopt`` restores it
        if self.offset and self.index_keys is not None:
            plain.keys, plain.values, plain.index_keys = self._contiguous(self.offset)
            # evaluated now: until then the gather holds views of the pool's buffers, and a write to them would copy
            # a whole buffer
            mx.eval(plain.keys, plain.values, plain.index_keys)
        return plain


class HybridAttentionCache(AttentionCache):
    """A served request's sparse-attention layer: contiguous buffers, as an ``AttentionCache``, and references to the
    full pages that hold its first ``src_len`` positions (``src_pages``: the entry it was resumed from, or its last
    stored copy). ``store_copy`` writes only the positions after them. ``layer`` is None only while
    ``copy_compact`` builds a copy."""

    def __init__(self, layer: int | None = None) -> None:
        super().__init__()
        self.layer = layer
        self.src_pages: list[int] = []
        self.src_len = 0
        self.index_from = 0         # raw index keys below this are zeros (resumed from an entry; never read)

    def __del__(self) -> None:
        # CPython runs this once the last reference goes (the request finished and its cache stored or dropped)
        pages, self.src_pages = getattr(self, "src_pages", []), []
        try:
            for page in pages:
                (self.__dict__.get("_owner") or _POOLS[self.layer]).unref(page)
        except Exception:  # noqa: BLE001 - interpreter shutdown, or no page was ever referenced
            pass

    @property
    def pool(self) -> KVPool:
        """The pool this cache's source pages belong to, fixed at first use (as ``PagedAttentionCache.pool``)."""

        owner = self.__dict__.get("_owner")
        if owner is None:
            owner = self.__dict__["_owner"] = pool(self.layer)
        return owner

    def _set_src(self, pages: list[int], length: int) -> None:
        for page in pages:
            self.pool.ref(page)
        for page in self.src_pages:
            self.pool.unref(page)
        self.src_pages, self.src_len = list(pages), length

    def trim(self, n: int, ratio: int = RATIO) -> int:
        n = super().trim(n, ratio)
        if self.index_from and self.offset // ratio * ratio < self.index_from:
            # the indexer would pool the next block from raw keys this cache does not hold (zeros): fail, not guess
            raise RuntimeError(f"TF_KV_PAGED: rollback to {self.offset} reaches below the raw index keys kept from "
                               f"{self.index_from}")
        if self.src_len > self.offset:              # positions past the rollback are written again: not the pages'
            keep = self.offset // PAGE
            self._set_src(self.src_pages[:keep], self.offset)
        return n

    def copy_compact(self, reserve: int = 0) -> "HybridAttentionCache":
        """``AttentionCache.copy_compact`` (a request's contiguous copy), referencing the same full pages."""

        clone = super().copy_compact(reserve)       # type(self)(): a HybridAttentionCache without a layer
        clone.layer, clone.index_from = self.layer, self.index_from
        clone.src_pages, clone.src_len = [], 0      # its references are taken below, never inherited
        length = min(self.src_len, self.offset)
        clone._set_src(self.src_pages[:length // PAGE], length)
        return clone

    def _plain_copy(self) -> AttentionCache:
        """A contiguous AttentionCache copy of this cache, as without TF_KV_PAGED. Evaluated here."""

        plain = AttentionCache()
        plain.keys, plain.values, plain.index_keys = self.keys, self.values, self.index_keys
        plain.pooled, plain.offset, plain.reserved = self.pooled, self.offset, self.reserved
        clone = plain.copy_compact()
        clone.index_from = self.index_from      # raw index keys below it are zeros; kept so ``adopt`` restores it
        mx.eval(*[a for a in (clone.keys, clone.values, clone.index_keys) if a is not None])
        return clone

    def snapshot_copy(self) -> AttentionCache:
        """This cache as a self-contained AttentionCache for a disk snapshot: its source pages are this process's
        references and must not be written out. Evaluated here."""

        return self._plain_copy()

    def store_copy(self) -> AttentionCache:
        """This cache's copy for the prompt cache (a prefill checkpoint, or a finished request's entry): a
        PagedAttentionCache that references the source pages and holds new pages for the positions after them. The new
        pages are written from evaluated temporaries: a lazy write whose source is a view of these buffers would keep
        them referenced, and MLX would copy them whole at their next write. The entry's full pages then become this
        cache's source, so its next stored copy writes only what follows. Evaluated here. With no page to be had, or
        more new pages than the pool holds, a contiguous copy."""

        n = self.offset
        entry = PagedAttentionCache(self.layer)
        entry.__dict__["_owner"] = self.pool        # its pages come from this cache's pool
        entry.offset, entry.reserved = n, self.reserved
        entry.pooled = None if self.pooled is None else mx.array(self.pooled)
        if n == 0 or self.keys is None:
            return entry
        p = self.pool
        if p.shape is None:
            if int(self.keys.shape[0]) != 1:
                raise RuntimeError("TF_KV_PAGED stores batch-1 caches only")
            p.shape = (int(self.keys.shape[1]), int(self.keys.shape[3]), self.keys.dtype)
        shared = min(self.src_len, n) // PAGE
        # more new pages than the pool can ever hold next to this cache's source pages: contiguous at once, so no
        # prompt-cache entry is evicted for a copy that cannot fit
        if -(-(n - shared * PAGE) // PAGE) > p.pages - len(set(self.src_pages)):
            print(f"[tensorfold] TF_KV_PAGED: {n} positions do not fit the {p.pages}-page pool of layer "
                  f"{self.layer}; this copy is contiguous", flush=True)
            return self._plain_copy()
        for page in self.src_pages[:shared]:
            p.ref(page)
            entry.pages.append(page)
        try:
            for start in range(shared * PAGE, n, PAGE):
                stop = min(n, start + PAGE)
                tk = mx.contiguous(self.keys[:, :, start:stop])
                tv = mx.contiguous(self.values[:, :, start:stop])
                mx.eval(tk, tv)
                page = p.alloc()
                entry.pages.append(page)
                p.write(page, tk, tv)
        except PoolFull as exc:
            del entry                               # its __del__ gives back the pages it took
            print(f"[tensorfold] TF_KV_PAGED: {exc}; this copy is contiguous", flush=True)
            return self._plain_copy()
        lo = _index_from(self.pooled, n)
        kept = self.index_keys[:, lo:n]
        entry.index_keys = kept * mx.array(1, kept.dtype)      # a new buffer, not a view of this cache's
        entry.index_from = lo
        mx.eval(entry.index_keys, *p.ks, *p.vs)
        self._set_src(entry.pages[:n // PAGE], n)
        return entry
