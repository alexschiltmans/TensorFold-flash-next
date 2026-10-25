"""The chat app behind TensorFold's OpenAI endpoint: one model, requests queued onto one engine thread.

A request is rendered with the model's chat template, queued, and decoded by the lane engine
(``engine.lane_engine.LaneEngine``; model families with their own forward take its family rounds,
``engine.lane_family``). Tokens stream back as they are committed: reasoning (inside the prompt's open think block) as
``reasoning_content``, tool calls as OpenAI tool_call deltas while they are written, the rest as content.

Prompts are cached: after a request, the conversation's prefix caches are kept (``CheckpointStore``), so a
follow-up that extends the same conversation only prefills its new suffix. The system block an agent client
sends with every session is saved to disk once and pinned, and the newest conversations are saved at shutdown
and read back on demand.

Sampling is exact: a token is a fixed function of its position's logits, the request's seed and the position
(``engine.exact_sampling``), so drafted rounds emit exactly what one-token-a-round decoding emits.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import itertools
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable
import uuid

from tensorfold.engine.lane_engine import LaneEngine, LaneStream, SuffixLookupProposer
from tensorfold.server.http import (
    CLIENT,
    RequestError,
    _normalize_tool_call_arguments,
    parse_harmony_output,
    served_model_ids,
    streaming_visible_text,
    strip_trailing_stops,
)


_THINK_END = "</think>"
_CALL_OPEN = "<tool_call>"
_CALL_CLOSE = "</tool_call>"


def _partial_tag(text: str, tag: str) -> int:
    """How many characters at the end of ``text`` could be the start of ``tag``."""

    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


def split_thinking(text: str, *, finished: bool) -> tuple[str, str]:
    """(reasoning, answer) of a reply whose prompt opened a think block (Qwen3.8 thinking).

    Everything before ``</think>`` is reasoning. While the reply streams, a tail that could
    begin ``</think>`` is held back, so what has been sent as reasoning never changes.
    """

    end = text.find(_THINK_END)
    if end < 0:
        return text[: len(text) - (0 if finished else _partial_tag(text, _THINK_END))], ""
    return text[:end], text[end + len(_THINK_END):].lstrip("\n")


def hide_tool_calls(text: str, *, finished: bool) -> str:
    """Reply text without its ``<tool_call>`` blocks, which reach the client as tool_call deltas.

    While streaming, an unclosed block and a tail that could begin ``<tool_call>`` are held
    back, so the visible text only ever grows (the markup reached a client as text, 2026-09-23).
    """

    out: list[str] = []
    pos = 0
    while True:
        start = text.find(_CALL_OPEN, pos)
        if start < 0:
            tail = text[pos:]
            out.append(tail[: len(tail) - (0 if finished else _partial_tag(tail, _CALL_OPEN))])
            return "".join(out)
        out.append(text[pos:start])
        end = text.find(_CALL_CLOSE, start)
        if end < 0:
            return "".join(out)
        pos = end + len(_CALL_CLOSE)


def render_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
    reasoning_effort: str | None = None,
) -> list[int]:
    """The same chat-template rendering the single-stream server performs.

    ``reasoning_effort`` (thinking on): Qwen3.8's template defaults to "xhigh", which writes an
    instruction at the top of the system prompt; "medium" writes none, so the system block (and
    its precomputed snapshot) is the same with thinking on or off.
    """

    messages = _normalize_tool_call_arguments(messages)
    kwargs: dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    if enable_thinking and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if tools:
        kwargs["tools"] = tools
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("tools", None)
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(rendered, str):
        rendered = tokenizer.encode(rendered)
    return [int(t) for t in rendered]


def eos_ids_of(tokenizer: Any) -> frozenset[int]:
    values = getattr(tokenizer, "eos_token_ids", None)
    if values:
        return frozenset(int(t) for t in values)
    single = getattr(tokenizer, "eos_token_id", None)
    return frozenset({int(single)}) if single is not None else frozenset()


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def common_prefix_len(a: Any, b: Any) -> int:
    """``longest_common_prefix`` for long token lists (numpy; a 131k-token prompt in a few ms)."""

    import numpy as np

    n = min(len(a), len(b))
    if n == 0:
        return 0
    x = np.asarray(a[:n], dtype=np.int64)
    y = np.asarray(b[:n], dtype=np.int64)
    differ = np.flatnonzero(x != y)
    return int(differ[0]) if differ.size else n


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return default if not value else value not in ("0", "false", "no", "off")


# Prefix checkpoints for long prompts (TF_PREFIX_ANCHORS=1). A stored prefix is worth its full length or nothing
# (GDN state cannot be truncated), and checkpoints were only taken at the rendered history boundary, the system
# block and the reply's end. A prompt that shares a long prefix with an earlier one but differs before those points
# (the same document with another question; a client that re-renders or edits the tail of a long conversation)
# therefore re-prefilled from zero: 64K prompts at about 310 s each, every time (Flash-Next, M4 Pro).
# llama.cpp keeps one KV and truncates it to the common prefix: it recomputes about 512 tokens past a checkpoint
# it takes before a prompt's last batch. Two checkpoints do the same here:
#   - a tail checkpoint TAIL_BACKOFF tokens before the end of a prompt that prefills at least TAIL_MIN_PREFILL
#     tokens, so a next prompt that differs only in its last TAIL_BACKOFF tokens reuses the rest;
#   - a divergence anchor ANCHOR_BACKOFF tokens before the end of the longest prefix a prompt shares with a recent
#     prompt or a stored entry, when that reuses ANCHOR_MIN_GAIN tokens more than its own hit: later prompts that
#     share the prefix then reuse all of it. The backoff keeps the anchor a prefix of prompts that diverge a few
#     tokens earlier ("Write a Python function" / "Write a Python context manager").
# With TF_PREFIX_SHARE the nested entries share one KV, so each extra checkpoint costs its recurrent state only.
ANCHOR_BACKOFF = 16
ANCHOR_MIN_GAIN = 256
# anchors evicted after the other conversation entries; older ones become ordinary entries (as pinned_slots)
ANCHOR_SLOTS = 2
TAIL_BACKOFF = 512
TAIL_MIN_PREFILL = 4096
# earlier prompts an anchor is looked for in (besides the stored entries' own tokens)
ANCHOR_RECENT = 4


def share_prefix_kv(older: list[Any], newer: list[Any]) -> int:
    """Point ``older``'s append-only KV arrays at ``newer``'s buffers where they hold the same values, so a stored
    prefix of a stored conversation costs its recurrent state only (TF_PREFIX_SHARE=1). Returns the bytes of the
    arrays now shared.

    Layer caches with ``offset`` and position-indexed ``keys``/``values`` (axis 2) and ``index_keys`` (axis 1)
    only read and write from ``offset`` on through these arrays, so ``older`` can hold ``newer``'s longer array:
    positions below ``older.offset`` are compared bit for bit before the swap (a prefix computed in another
    chunking may differ). MLX never writes a buffer another array still references, so a cache resumed from either
    entry copies on its first write, as it does from an unshared entry."""

    import mlx.core as mx

    if len(older) != len(newer):
        return 0
    found: list[tuple[Any, str, Any, int]] = []
    checks: list[Any] = []
    for a, b in zip(older, newer):
        if type(a) is not type(b) or callable(getattr(a, "page_refs", None)):
            continue                # a paged layer shares its pages already (its index keys need not start at 0)
        n = getattr(a, "offset", None)
        m = getattr(b, "offset", None)
        if not isinstance(n, int) or not isinstance(m, int) or not 0 < n <= m:
            continue
        for name, axis in (("keys", 2), ("values", 2), ("index_keys", 1)):
            x, y = getattr(a, name, None), getattr(b, name, None)
            if not isinstance(x, mx.array) or not isinstance(y, mx.array) or x is y:
                continue
            if x.ndim != y.ndim or x.ndim <= axis or x.dtype != y.dtype or min(x.shape[axis], y.shape[axis]) < n:
                continue
            if any(x.shape[i] != y.shape[i] for i in range(x.ndim) if i != axis):
                continue
            cut = (slice(None),) * axis + (slice(0, n),)
            checks.append(mx.array_equal(x[cut], y[cut]))
            found.append((a, name, y, int(x.nbytes)))
    if not found:
        return 0
    if os.environ.get("TF_CACHE_COMPACT", "").strip() not in ("", "0"):
        # one comparison at a time: each builds a boolean array half its KV's size, and evaluating them all at
        # once held every one of them (several GB at 32K with a few entries; 2-4 GB wired spikes)
        same = [bool(check.item()) for check in checks]
    else:
        same = mx.stack(checks).tolist()
    shared = 0
    for (a, name, y, nbytes), equal in zip(found, same):
        if equal:
            setattr(a, name, y)
            shared += nbytes
    return shared


def unique_cache_bytes(caches: Any) -> int:
    """Bytes of the distinct MLX arrays the given cache lists hold (arrays shared between entries count once). A layer
    cache that holds pages of a shared pool (``page_refs()``, ``page_bytes()``: TF_KV_PAGED) counts each page once."""

    import mlx.core as mx

    seen: dict[Any, int] = {}
    for cache in caches:
        for item in cache:
            refs = getattr(item, "page_refs", None)
            if callable(refs):
                size = int(item.page_bytes())
                for ref in refs():
                    seen[("page", ref)] = size
            for value in vars(item).values():
                values = value if isinstance(value, (list, tuple)) else (value,)
                for v in values:
                    if isinstance(v, mx.array):
                        seen[id(v)] = int(v.nbytes)
    return sum(seen.values())


def choose_checkpoints(
    history_len: int, cached: int, last_prompt: list[int] | None, prompt: list[int]
) -> list[int]:
    """Where to snapshot this request's prefill for the NEXT turn.

    The rendered history boundary is the natural candidate: the generation
    prompt's tail does not survive re-rendering, the history does. When the
    same conversation's previous prompt shares most but not all of that
    history, the client may be appending a per-turn block (environment,
    timestamps, tool results) that will not be there next turn, so the
    stable prefix is a second candidate, as in the single-stream server's
    clamp. Both are kept: with an empty-think-block template the stable
    prefix is merely last turn's boundary and falls inside the reused part,
    while with a per-turn block only the stable prefix will match. Nothing
    inside the reused part, nothing at or past the prompt's end.
    """

    candidates = {int(history_len)}
    if last_prompt:
        stable = longest_common_prefix(last_prompt, prompt)
        if 0 < stable < int(history_len) and stable >= int(history_len) // 2:
            candidates.add(stable)
    return sorted(at for at in candidates if int(cached) < at < len(prompt))


def is_title_request(messages: list[dict[str, Any]], tools: Any) -> bool:
    """A client's session-title request: a short system prompt about a title, no tools, short messages.

    Agent clients send one ("Write a ~5 word title ...") with a new session's first turn, 10 ms or so
    before the turn itself. It runs in the background: it gives the turn a head start, goes after
    any other waiting request, and is preempted when one arrives before its first token (after that it
    finishes: a rerun would not reproduce the tokens already sent).
    """

    if tools or not messages or messages[0].get("role") != "system":
        return False
    size = sum(len(m["content"]) if isinstance(m.get("content"), str) else 4096 for m in messages)
    text = messages[0].get("content")
    return size < 4096 and isinstance(text, str) and "title" in text.lower()



class IncrementalText:
    """The text of a growing token list, decoding only its newest tokens (with the previous chunk as context,
    so multi-byte characters split across tokens come out whole). Decoding the whole reply every round
    cost 5.8 ms a round at 32k tokens, under the tokenizer lock the tool-call proposer also takes."""

    def __init__(self, tokenizer: Any, lock: Any) -> None:
        self.tokenizer = tokenizer
        self.lock = lock
        self.tokens: list[int] = []
        self.text = ""
        self._prefix = 0
        self._read = 0

    def extend(self, tokens: list[int]) -> str:
        self.tokens.extend(int(t) for t in tokens)
        with self.lock:
            before = self.tokenizer.decode(self.tokens[self._prefix:self._read])
            after = self.tokenizer.decode(self.tokens[self._prefix:])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            self.text += after[len(before):]
            self._prefix, self._read = self._read, len(self.tokens)
        return self.text



_REQUEST = threading.local()


def _mlx_version() -> str:
    import mlx.core as mx

    return str(mx.__version__)


def _draft_kinds(stream: Any) -> str:
    """The done line's per-proposer draft counts (lane family): copy/head drafted and accepted, head rounds by depth."""

    kinds = getattr(stream, "__dict__", {}).get("draft_kinds") if stream is not None else None
    if not kinds:
        return ""
    parts = [f"{k}={v[1]}/{v[0]}" for k, v in kinds.items() if isinstance(v, list)]
    depths = sorted((k, v) for k, v in kinds.items() if k.startswith("h") and isinstance(v, int))
    if depths:
        parts.append("depths=" + ",".join(f"{k[1:]}:{v}" for k, v in depths))
    return "kinds " + " ".join(parts) + " "


def _token_sha(tokens: list[int]) -> str:
    """A short fingerprint of a reply's tokens: two runs match byte for byte iff these match."""

    import hashlib

    return hashlib.sha256(",".join(str(int(t)) for t in tokens).encode()).hexdigest()[:12]


class _LockedTokenizer:
    """The app's tokenizer behind its lock, for proposers on the scheduler thread."""

    def __init__(self, tokenizer: Any, lock: Any) -> None:
        self._tokenizer = tokenizer
        self._lock = lock

    def decode(self, ids: list[int]) -> str:
        with self._lock:
            return self._tokenizer.decode(ids)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        with self._lock:
            return self._tokenizer.encode(text, **kwargs)

    def convert_tokens_to_ids(self, token: str) -> Any:
        with self._lock:
            return self._tokenizer.convert_tokens_to_ids(token)


@dataclass
class CheckpointEntry:
    tokens: list[int]
    cache: list[Any]
    last_prompt: list[int]
    nbytes: int = 0
    # A system block (loaded from disk, or saved to it): outside the slot count.
    pinned: bool = False
    # A divergence anchor (TF_PREFIX_ANCHORS): evicted after the other conversation entries.
    anchor: bool = False
    # A tail checkpoint (TF_PREFIX_ANCHORS: TAIL_BACKOFF before a long prompt's end): the newest one of the
    # conversation used last is evicted after the other unpinned entries (TF_PREFIX_KEEP_TAIL)
    tail: bool = False


def save_conversations(store: "CheckpointStore", directory: Path, model_id: str, *, keep: int = 2,
                       limit_bytes: int = 10 * 1024**3) -> int:
    """Write the store's most recently used conversation entries to ``directory``; returns how many.

    A restart threw away the conversation the user was in: the agent's next turn, at 70,510
    tokens, then prefilled for about three minutes (2026-09-23). Saved at shutdown (~1 s
    a 4.5 GB entry) and read back only when a prompt starts with one, they make a restart
    cost that turn a second. System blocks are not written here (they have their own
    directory); at most ``keep`` files stay, the rest are deleted.
    """

    from tensorfold.engine.prefix_snapshots import save_snapshot

    with store._lock:
        entries = [entry for entry in store._entries if not entry.pinned]   # most recently used first
    # longest first (stable, so recency breaks ties): a long conversation costs minutes to prefill again and a
    # short one about a second, and background requests (data generation, benchmarks) must not push the
    # user's conversation out: 2026-09-24 a restart after test traffic re-prefilled a 70,510-token agent turn (128 s)
    entries.sort(key=lambda entry: -len(entry.tokens))
    saved = total = 0
    for entry in entries:
        if saved >= keep or total + entry.nbytes > limit_bytes:
            break
        started = time.perf_counter()
        try:
            save_snapshot(directory, model_id, entry.tokens, entry.cache, keep=keep)
        except Exception as exc:  # noqa: BLE001 - a full disk must not hang the shutdown
            print(f"[lanes] conversation save failed: {type(exc).__name__}: {exc}", flush=True)
            break
        saved += 1
        total += entry.nbytes
        print(f"[lanes] saved conversation checkpoint tokens={len(entry.tokens)} "
              f"({entry.nbytes / 1024**3:.1f} GiB) in {time.perf_counter() - started:.1f}s", flush=True)
    return saved


def newest_tail(entries: list[CheckpointEntry]) -> CheckpointEntry | None:
    """The tail checkpoint of the conversation used last (TF_PREFIX_KEEP_TAIL), or None. Its next request may re-ask or
    edit the last message, and then its tail is the only stored prefix it can hit (a 188K prompt prefilled cold for
    871 s). Only the longest such tail is protected: the newest one."""

    head = next((entry for entry in entries if not entry.pinned), None)
    if head is None:
        return None
    best: CheckpointEntry | None = None
    for entry in entries:
        if (entry.tail and not entry.pinned and len(entry.tokens) <= len(head.tokens)
                and head.tokens[: len(entry.tokens)] == entry.tokens):
            if best is None or len(entry.tokens) > len(best.tokens):
                best = entry
    return best


def _entry_kind(entry: CheckpointEntry) -> str:
    return "tail" if entry.tail else "anchor" if entry.anchor else "pinned" if entry.pinned else "entry"


def _evicted_line(tokens: int, reason: str, kind: str, newest: bool, nbytes: int, left: int,
                  freed: int | None = None, pages: bool = False) -> str:
    """The log line for one eviction; ``freed`` is the drop in the store's bytes, when it was computed."""

    where = " for KV pages" if pages else ""
    freed_part = "" if freed is None else f" freed={freed / 1024**3:.2f} GiB"
    return (f"[tensorfold] prompt cache: evicted a {tokens}-token entry{where} (reason={reason} kind={kind} "
            f"newest_tail={'yes' if newest else 'no'} entry={nbytes / 1024**3:.2f} GiB{freed_part} "
            f"left={left} entries)")


class CheckpointStore:
    """LRU of absorbed conversation prefixes with their single-row caches.

    A hit must be a strict prefix of the new prompt: GDN recurrent state
    cannot be truncated, so a stored prefix is worth its full length or
    nothing. Up to three entries per turn are kept (the history boundary, a
    clamped stable prefix, the absorbed end of the reply) because which one
    the next turn matches depends on how the client re-renders the assistant
    turn. Eviction is least-recently-used under BOTH a slot count and a byte
    budget: a 27B lane holds about 64 KB of KV per token plus 150 MB of GDN
    state, so long agent contexts are what the budget is for.

    System blocks are pinned: they sit outside the slot count and the budget
    takes conversation entries first. With three slots an agent client's
    title request pushed the 21,415- and 20,903-token blocks out before the
    session's first turn arrived, which then paid 2,114 tokens of prefill
    (2026-09-23). At most ``pinned_slots`` stay pinned; older ones become
    ordinary entries.
    """

    def __init__(
        self,
        slots: int,
        copier: Callable[[list[Any]], list[Any]],
        *,
        budget_bytes: int | None = None,
        sizer: Callable[[list[Any]], int] | None = None,
        pinned_slots: int = 3,
        share_prefixes: bool = False,
        keep_tail: bool = True,
    ) -> None:
        if slots < 1:
            raise ValueError("slots must be positive")
        if budget_bytes is not None and budget_bytes < 1:
            raise ValueError("budget_bytes must be positive when given")
        self.slots = int(slots)
        self.pinned_slots = int(pinned_slots)
        self.copier = copier
        self.budget_bytes = None if budget_bytes is None else int(budget_bytes)
        self.sizer = sizer
        # TF_PREFIX_SHARE: a stored prefix of another entry holds that entry's KV arrays (``share_prefix_kv``), and
        # the budget counts distinct arrays
        self.share_prefixes = bool(share_prefixes)
        # TF_PREFIX_KEEP_TAIL: the newest conversation's tail checkpoint is evicted after the other entries
        self.keep_tail = bool(keep_tail)
        self._entries: list[CheckpointEntry] = []
        # reentrant, so evict_lru (which a full page pool calls) is safe in code that already holds it
        self._lock = threading.RLock()
        # entries hold pages of a shared pool (``page_refs``): the budget counts distinct pages and arrays
        self._paged = False
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _bytes(self, entries: list[CheckpointEntry]) -> int:
        if self.share_prefixes or self._paged:
            return unique_cache_bytes(entry.cache for entry in entries)
        return sum(entry.nbytes for entry in entries)

    @property
    def nbytes(self) -> int:
        return self._bytes(self._entries)

    def match(self, prompt: list[int], reserve: int = 0) -> tuple[int, list[Any], list[int]] | None:
        """Longest strict-prefix hit as (length, cache copy, previous prompt); with ``reserve`` the copy has room
        for that many positions (TF_CACHE_COMPACT: one allocation instead of a copy and a regrow)."""

        with self._lock:
            best: CheckpointEntry | None = None
            for entry in self._entries:
                tokens = entry.tokens
                if 0 < len(tokens) < len(prompt) and prompt[: len(tokens)] == tokens:
                    if best is None or len(tokens) > len(best.tokens):
                        best = entry
            if best is None:
                self.misses += 1
                return None
            self.hits += 1
            self._entries.remove(best)
            self._entries.insert(0, best)
            previous = list(best.last_prompt)
            best.last_prompt = list(prompt)
            copy = self.copier(best.cache, reserve) if reserve else self.copier(best.cache)
            return len(best.tokens), copy, previous

    def longest(self, prompt: list[int]) -> int:
        """Length of the longest strict-prefix entry (0 if none); not counted as a hit or miss."""

        with self._lock:
            return max((len(entry.tokens) for entry in self._entries
                        if 0 < len(entry.tokens) < len(prompt) and prompt[: len(entry.tokens)] == entry.tokens),
                       default=0)

    def insert(self, tokens: list[int], cache: list[Any], *, last_prompt: list[int],
               pinned: bool = False, anchor: bool = False, tail: bool = False) -> None:
        if not tokens:
            return
        # a layer cache with ``store_copy()`` is stored in that form: with TF_KV_PAGED a finished request's contiguous
        # buffers become pages shared with the entry it was resumed from (the caller drops the original)
        if any(callable(getattr(item, "store_copy", None)) for item in cache):
            cache = [item.store_copy() if callable(getattr(item, "store_copy", None)) else item for item in cache]
        paged = any(callable(getattr(item, "page_refs", None)) for item in cache)
        if paged:
            self._paged = True
            nbytes = unique_cache_bytes([cache])     # ``state`` leaves the pages out
        else:
            nbytes = int(self.sizer(cache)) if self.sizer is not None else 0
        with self._lock:
            replaced = [entry for entry in self._entries if entry.tokens == list(tokens)]
            kept = [entry for entry in self._entries if entry.tokens != list(tokens)]
            pinned = pinned or any(entry.pinned for entry in replaced)
            anchor = anchor or any(entry.anchor for entry in replaced)
            tail = tail or any(entry.tail for entry in replaced)
            entry = CheckpointEntry(list(tokens), cache, list(last_prompt), nbytes, pinned, anchor, tail=tail)
            if self.share_prefixes:
                # stored prefixes of the new entry (at least half its length: a short system block must not keep a
                # long conversation's arrays alive) take its KV arrays; stored entries it is a prefix of lend theirs
                for other in kept:
                    a, b = (other, entry) if len(other.tokens) < len(entry.tokens) else (entry, other)
                    if 2 * len(a.tokens) >= len(b.tokens) and b.tokens[: len(a.tokens)] == a.tokens:
                        try:
                            share_prefix_kv(a.cache, b.cache)
                        except Exception as exc:  # noqa: BLE001 - sharing only saves memory
                            print(f"[tensorfold] prefix share failed: {type(exc).__name__}: {exc}", flush=True)
            entries = [entry, *kept]
            for extra in [e for e in entries if e.pinned][self.pinned_slots:]:
                extra.pinned = False
            for extra in [e for e in entries if e.anchor][ANCHOR_SLOTS:]:
                extra.anchor = False
            # the new entry itself is never evicted: least recently used first, conversations
            # before divergence anchors before the newest conversation's tail before system blocks
            newest_known = False
            newest: CheckpointEntry | None = None
            keep: CheckpointEntry | None = None
            events: list[dict[str, Any]] = []     # one per eviction; "freed" is filled in by the next pass
            before: int | None = None             # the store's bytes before the last drop
            while True:
                over_slots = sum(1 for e in entries if not e.pinned) > self.slots
                total = self._bytes(entries) if self.budget_bytes is not None and len(entries) > 1 else None
                if events and events[-1]["freed"] is None and total is not None and before is not None:
                    events[-1]["freed"] = before - total
                over_budget = total is not None and total > self.budget_bytes
                if not (over_slots or over_budget):
                    break
                if not newest_known:
                    newest = newest_tail(entries)
                    keep = newest if self.keep_tail else None
                    newest_known = True
                candidates = (
                    [i for i in range(1, len(entries))
                     if not entries[i].pinned and not entries[i].anchor and entries[i] is not keep]
                    or [i for i in range(1, len(entries)) if not entries[i].pinned and entries[i] is not keep]
                    or [i for i in range(1, len(entries)) if not entries[i].pinned]
                )
                if candidates:
                    dropped = entries.pop(candidates[-1])
                elif over_budget:
                    dropped = entries.pop()
                else:
                    break
                self.evictions += 1
                reason = "slots+budget" if over_slots and over_budget else "slots" if over_slots else "budget"
                events.append({
                    "tokens": len(dropped.tokens), "reason": reason, "kind": _entry_kind(dropped),
                    "newest": dropped is newest, "nbytes": dropped.nbytes, "left": len(entries), "freed": None,
                })
                # a dropped newest or kept entry is no longer in the store: let go of it now, so its pages return now
                if dropped is newest:
                    newest = None
                if dropped is keep:
                    keep = None
                del dropped
                before = total
            self._entries = entries
            lines = [_evicted_line(**event) for event in events]
        for line in lines:
            print(line, flush=True)

    def evict_lru(self) -> bool:
        """Drop the least recently used entry that holds pages, conversations before divergence anchors before the
        newest conversation's tail, for a model whose entries hold pages of a pool with no free page left
        (``attach_prompt_cache``). The entry last used goes too when nothing else is left; pinned system blocks never
        do, and entries without pages (contiguous copies) stay, since dropping them frees no page (the copy that needs
        the page is made contiguous instead). False when no unpinned entry holds pages."""

        import gc

        def holds_pages(entry: CheckpointEntry) -> bool:
            return any(callable(getattr(item, "page_refs", None)) and item.page_refs() for item in entry.cache)

        with self._lock:
            entries = self._entries
            paged = [i for i, e in enumerate(entries) if not e.pinned and holds_pages(e)]
            if not paged:
                return False
            newest = newest_tail(entries)
            keep = newest if self.keep_tail else None
            order = ([i for i in paged if not entries[i].anchor and entries[i] is not keep]
                     or [i for i in paged if entries[i] is not keep] or paged)
            dropped = entries.pop(order[-1])
            self.evictions += 1
            tokens, kind, nbytes = len(dropped.tokens), _entry_kind(dropped), dropped.nbytes
            is_newest, left = dropped is newest, len(entries)
            newest = keep = None      # a local reference here would keep the dropped entry's pages alive
        # the layers give their pages back in __del__ (CPython frees the entry when this last reference goes);
        # collect in case a reference cycle holds it
        del dropped
        gc.collect()
        print(_evicted_line(tokens, "pages", kind, is_newest, nbytes, left, pages=True), flush=True)
        return True

    def __len__(self) -> int:
        return len(self._entries)



@dataclass
class ChatJob:
    job_id: str
    prompt_ids: list[int]
    max_tokens: int
    temperature: float
    history_len: int = 0
    # Extra prefill snapshot points that other conversations can reuse: the end of the system-and-tools block.
    # An agent client sends the same ~20k-token block with every session; the history boundary alone never
    # matches a new session because it already contains that session's first message.
    shared_prefix_lens: tuple[int, ...] = ()
    # the request's proposer (suffix copies, a drafter model, tool-call structure); None: the engine's default
    proposer: Any = None
    # False ("draft": false): one token a round and no drafts, the serial reference output is checked against
    drafts: bool = True
    # exact_sampling.Sampling for this request (None = greedy)
    sampling: Any = None
    # thinking budget (0: none) and the tokens that close a think block ("\n</think>\n\n"; LaneStream)
    think_budget: int = 0
    think_close: tuple[int, ...] = ()
    think_end: int = -1
    # A client's side request (a session title): admitted after every other waiting job, and stopped
    # ("preempted") when one arrives before it has produced a token; its caller runs it again from the start.
    background: bool = False
    preempted: bool = False
    submitted_at: float = field(default_factory=time.perf_counter)
    started_at: float = 0.0
    prefilled_at: float = 0.0
    finished_at: float = 0.0
    cached_tokens: int = 0
    stream: LaneStream | None = None
    error: BaseException | None = None
    # the client hung up: not started if still waiting, stopped at the next round if running
    cancelled: bool = False
    chunks: "queue.Queue[list[int] | None]" = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)


class _JobQueue(queue.PriorityQueue):
    """Jobs in arrival order, background jobs after every other job."""

    def __init__(self) -> None:
        super().__init__()
        self._order = itertools.count()

    def put(self, job: ChatJob, block: bool = True, timeout: float | None = None) -> None:
        super().put((1 if job.background else 0, next(self._order), job), block, timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> ChatJob:
        return super().get(block, timeout)[2]

    def foreground_waiting(self) -> bool:
        with self.mutex:
            return bool(self.queue) and self.queue[0][0] == 0


class Scheduler:
    """Owns the engine on one thread: admits jobs, steps rounds, delivers tokens."""

    def __init__(
        self,
        engine: Any,
        *,
        lanes: int,
        eos_ids: frozenset[int],
        checkpoints: CheckpointStore | None = None,
        proposer_factory: Callable[[], Any] | None = None,
        idle_wait: float = 0.02,
        snapshot_dir: Any = None,
        session_dir: Any = None,
        model_id: str = "",
        anchors: bool = False,
    ) -> None:
        if lanes < 1:
            raise ValueError("lanes must be positive")
        # TF_PREFIX_ANCHORS: checkpoints where a prompt diverges from an earlier one (ANCHOR_BACKOFF)
        self.anchors = bool(anchors)
        self._recent_prompts: deque[list[int]] = deque(maxlen=ANCHOR_RECENT)
        self.snapshot_dir = snapshot_dir
        self.session_dir = session_dir
        self.model_id = model_id
        self.disk_blocks: Any = None
        self.session_blocks: Any = None
        if model_id and (snapshot_dir is not None or session_dir is not None):
            from tensorfold.engine.prefix_snapshots import DiskBlocks

            if snapshot_dir is not None:
                self.disk_blocks = DiskBlocks(Path(snapshot_dir), model_id)
            if session_dir is not None:
                self.session_blocks = DiskBlocks(Path(session_dir), model_id)
        self.engine = engine
        self.lanes = int(lanes)
        self.eos_ids = eos_ids
        self.checkpoints = checkpoints
        self.proposer_factory = proposer_factory
        self.idle_wait = float(idle_wait)
        self.slow_round_ms = 1000.0
        self._held: ChatJob | None = None
        self._queue = _JobQueue()
        self.preemptions = 0
        self._jobs: dict[str, ChatJob] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tensorfold-engine", daemon=True)
        self.rounds = 0
        self.failed_rounds = 0
        self.completed = 0
        self.starts = 0
        self._starting: ChatJob | None = None
        # run by the scheduler thread as it stops: its caches' arrays live on its own streams, so they
        # can only be evaluated (and saved) there ("There is no Stream(gpu, 2) in current thread")
        self.on_stop: Callable[[], Any] | None = None
        self.stall_s = 120.0            # no round, start or finish while requests wait: dump stacks
        self.stall_prefill_s = 900.0    # the same while one prefill runs (70k tokens: ~3 min)
        self._watchdog = threading.Thread(target=self._watch, name="tensorfold-watchdog", daemon=True)

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()
        if not self._watchdog.is_alive():
            self._watchdog.start()

    def _watch(self) -> None:
        """Print the queue and every thread's stack once if requests wait while nothing moves (a stall is
        otherwise invisible). ``kill -USR1 <pid>`` dumps the same stacks on demand."""

        from tensorfold.server.logsink import dump_stacks

        mark: tuple[int, int, int] | None = None
        since = time.perf_counter()
        dumped = False
        while not self._stop.wait(1.0):
            waiting = self._queue.qsize() + (self._held is not None) + len(self._jobs) + (self._starting is not None)
            now_mark = (self.rounds, self.starts, self.completed)
            now = time.perf_counter()
            if now_mark != mark or not waiting:
                mark, since, dumped = now_mark, now, False
                continue
            limit = self.stall_prefill_s if self._starting is not None else self.stall_s
            if now - since > limit and not dumped:
                dumped = True
                dump_stacks(f"[tensorfold] stalled {now - since:.0f}s: queued={self._queue.qsize()} "
                            f"held={self._held is not None} jobs={len(self._jobs)} "
                            f"active={self.engine.active_count} starting={self._starting is not None}; "
                            "every thread's stack follows")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def submit(self, job: ChatJob) -> None:
        if self._stop.is_set():
            raise RuntimeError("the scheduler is closed")
        self._queue.put(job)

    @property
    def active(self) -> int:
        return len(self._jobs)

    @staticmethod
    def finish_job(job: ChatJob, reason: str = "stop") -> None:
        """Ask the engine to stop a stream at the next round (safe from any thread)."""

        stream = job.stream
        if stream is not None and not stream.finished:
            stream.finish_reason = reason
            stream.finished = True
            stream.finished_at = time.perf_counter()

    # -- the loop -------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._loop()
        finally:
            if self.on_stop is not None:
                try:
                    self.on_stop()
                except Exception as exc:  # noqa: BLE001 - the shutdown must finish
                    print(f"[tensorfold] shutdown hook failed: {type(exc).__name__}: {exc}", flush=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._preempt_background()
            self._admit()
            self._starting = None
            self._retire_externally_finished()
            if self.engine.active_count == 0:
                if self._held is None:
                    try:
                        self._held = self._queue.get(timeout=self.idle_wait)
                    except queue.Empty:
                        continue
                continue  # _admit starts it
            try:
                landed = self.engine.step()
            except Exception as exc:  # noqa: BLE001 - one bad round must not kill the server
                self.failed_rounds += 1
                for job in list(self._jobs.values()):
                    job.error = exc
                    self._finish(job)
                self._jobs.clear()
                self.engine.reset()
                continue
            self.rounds += 1
            if self.engine.round_stats:
                last = self.engine.round_stats[-1]
                if last.total_ms > self.slow_round_ms:
                    print(f"[tensorfold] slow round {last.total_ms:.0f} ms streams={last.streams} "
                          f"width={last.width} rows={last.rows} forward={last.forward_ms:.0f}", flush=True)
                if len(self.engine.round_stats) > 4096:
                    del self.engine.round_stats[:2048]
            for stream_id, tokens in landed.items():
                job = self._jobs.get(stream_id)
                if job is None:
                    continue
                if tokens:
                    job.chunks.put(list(tokens))
                if job.stream is not None and job.stream.finished:
                    del self._jobs[stream_id]
                    self._retire(job)

    def _preempt_background(self) -> None:
        """A waiting request takes the engine from a running background job at the next round, as long as that job
        has not produced a token yet. Stopped here, it runs again from the start once the engine is free. A job that
        has produced tokens runs to its end: a rerun does not reproduce them bit for bit (its logits differ in the
        low bits, batched or alone, cached or fresh prefill), and its caller would splice two different replies.
        Tokens are committed on this thread (engine.step), so no token reaches the caller after this check."""

        if not self._queue.foreground_waiting() and not (self._held is not None and not self._held.background):
            return
        for job in self._jobs.values():
            if (job.background and not job.preempted and job.stream is not None and not job.stream.finished
                    and not job.stream.emitted):
                job.preempted = True
                self.preemptions += 1
                self.finish_job(job, reason="preempted")

    def _retire_externally_finished(self) -> None:
        for stream_id, job in list(self._jobs.items()):
            if job.stream is not None and job.stream.finished:
                del self._jobs[stream_id]
                self._retire(job)

    def _admit(self) -> None:
        """Start waiting jobs while there are free lanes; while rounds run, one prefill a round (a burst of
        arrivals would otherwise sit through back-to-back prefills before any of them streams)."""

        admitted = 0
        while self.engine.active_count < self.lanes:
            job = self._held
            self._held = None
            if job is not None and job.background and self._queue.foreground_waiting():
                self._queue.put(job)          # a request that arrived since goes first
                job = None
            if job is None:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    return
            if job.cancelled:
                self._finish(job)
                continue
            if admitted and self.engine.active_count > 0:
                self._held = job
                return
            self._start_job(job)
            admitted += 1

    def _read_disk_block(self, prompt: list[int]) -> None:
        """Put the longest stored prefix of this prompt in the store, if the store has less. System blocks
        (pinned) and the conversations saved at the last shutdown (ordinary entries) are both read on demand."""

        if self.checkpoints is None or (self.disk_blocks is None and self.session_blocks is None):
            return
        try:
            have = self.checkpoints.longest(prompt)
            found: Any = None
            for blocks, pinned in ((self.disk_blocks, True), (self.session_blocks, False)):
                hit = None if blocks is None else blocks.best(prompt, have if found is None else len(found[1]))
                if hit is not None:
                    found = (hit[0], hit[1], blocks, pinned)
            if found is None:
                return
            from tensorfold.engine.prefix_snapshots import load_snapshot

            started = time.perf_counter()
            loaded = load_snapshot(found[0], self.model_id)
            if loaded is None:
                return
            tokens, cache = loaded
            self.checkpoints.insert(tokens, cache, last_prompt=tokens, pinned=found[3])
            found[2].touch(tokens)
            print(f"[tensorfold] read {'system-block' if found[3] else 'conversation'} snapshot tokens={len(tokens)} "
                  f"from disk in {time.perf_counter() - started:.2f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - a bad file costs a prefill, never the request
            print(f"[tensorfold] snapshot read failed: {type(exc).__name__}: {exc}", flush=True)

    def _start_job(self, job: ChatJob) -> None:
        job.started_at = time.perf_counter()
        self.starts += 1
        self._starting = job
        try:
            cache = None
            cached = 0
            last_prompt: list[int] | None = None
            checkpoints_at: list[int] = []
            if self.checkpoints is not None:
                self._read_disk_block(job.prompt_ids)
                from tensorfold.engine.lane_family import reply_room

                hit = self.checkpoints.match(job.prompt_ids, reserve=reply_room(len(job.prompt_ids), job.max_tokens))
                if hit is not None:
                    cached, cache, last_prompt = hit
                    if self.disk_blocks is not None and cached in job.shared_prefix_lens:
                        self.disk_blocks.touch(job.prompt_ids[:cached])
                checkpoints_at = sorted({
                    *choose_checkpoints(job.history_len, cached, last_prompt, job.prompt_ids),
                    *(n for n in job.shared_prefix_lens if cached < n < len(job.prompt_ids)),
                })
            anchor_at, tail_at = self._anchor(job.prompt_ids, cached, checkpoints_at)
            checkpoints_at = sorted({*checkpoints_at, *(at for at in (anchor_at, tail_at) if at)})
            proposer = job.proposer
            if proposer is None and job.drafts and self.proposer_factory is not None:
                proposer = self.proposer_factory()
            stream = LaneStream(
                stream_id=job.job_id,
                prompt_ids=list(job.prompt_ids),
                max_new_tokens=int(job.max_tokens),
                eos_ids=self.eos_ids,
                proposer=proposer if job.drafts else None,
                drafts=bool(job.drafts),
                sampling=job.sampling,
                think_budget=int(job.think_budget),
                think_close=tuple(job.think_close),
                think_end=int(job.think_end),
                think_open=bool(job.think_budget),
            )
            job.stream = stream
            self.engine.add_stream(stream, cache=cache, cached_tokens=cached, checkpoints_at=checkpoints_at)
            if self.checkpoints is not None:
                for tokens, snapshot in stream.history_checkpoints:
                    shared = len(tokens) in job.shared_prefix_lens
                    anchored = bool(anchor_at) and len(tokens) == anchor_at and not shared
                    tailed = bool(tail_at) and len(tokens) == tail_at and not shared
                    self.checkpoints.insert(tokens, snapshot, last_prompt=job.prompt_ids, pinned=shared,
                                            anchor=anchored, tail=tailed)
                    if anchored:
                        print(f"[tensorfold] anchor checkpoint tokens={len(tokens)} (prompt={len(job.prompt_ids)} "
                              f"cached={cached}: shares this prefix with an earlier prompt)", flush=True)
                    if tailed:
                        print(f"[tensorfold] tail checkpoint tokens={len(tokens)} (prompt={len(job.prompt_ids)} "
                              f"cached={cached})", flush=True)
                    if self.snapshot_dir is not None and shared:
                        self._persist(tokens, snapshot)
            stream.history_checkpoints = []
            job.prefilled_at = time.perf_counter()
            job.cached_tokens = int(cached)
            if stream.emitted:
                job.chunks.put(list(stream.emitted))
            if stream.finished:
                self._retire(job)
            else:
                self._jobs[stream.stream_id] = job
        except Exception as exc:  # noqa: BLE001 - reported to the waiting request
            job.error = exc
            print(f"[tensorfold] start failed {job.job_id} cached={job.cached_tokens}: {type(exc).__name__}: {exc}",
                  flush=True)
            self._finish(job)

    def _anchor(self, prompt: list[int], cached: int, checkpoints_at: list[int]) -> tuple[int, int]:
        """(divergence anchor, tail checkpoint) positions for this prompt, 0 for none (TF_PREFIX_ANCHORS; see
        ANCHOR_BACKOFF). Remembers the prompt for the next request."""

        if not self.anchors or self.checkpoints is None:
            return 0, 0
        at = tail = 0
        cached = int(cached)
        try:
            if len(prompt) - cached > ANCHOR_MIN_GAIN + ANCHOR_BACKOFF:
                with self.checkpoints._lock:
                    others = [entry.tokens for entry in self.checkpoints._entries]
                shared = max((common_prefix_len(prompt, other) for other in [*self._recent_prompts, *others]),
                             default=0)
                candidate = min(shared, len(prompt) - 1) - ANCHOR_BACKOFF
                if candidate - cached >= ANCHOR_MIN_GAIN and candidate not in checkpoints_at:
                    at = candidate
            candidate = len(prompt) - TAIL_BACKOFF
            if (len(prompt) - cached >= TAIL_MIN_PREFILL and candidate > max(cached, at)
                    and all(abs(candidate - b) >= ANCHOR_MIN_GAIN for b in checkpoints_at)):
                tail = candidate
        except Exception as exc:  # noqa: BLE001 - an anchor only saves a later prefill
            print(f"[tensorfold] anchor search failed: {type(exc).__name__}: {exc}", flush=True)
        self._recent_prompts.append(list(prompt))
        return at, tail

    def _persist(self, tokens: list[int], cache: list[Any]) -> None:
        """Write a system-block snapshot to disk once, so a restart does not lose it."""

        from tensorfold.engine.prefix_snapshots import save_snapshot

        try:
            started = time.perf_counter()
            path = save_snapshot(self.snapshot_dir, self.model_id, tokens, cache)
            if path is not None:
                print(f"[tensorfold] saved system-block snapshot tokens={len(tokens)} "
                      f"in {time.perf_counter() - started:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - a full disk must not fail the request
            print(f"[tensorfold] snapshot save failed: {type(exc).__name__}: {exc}", flush=True)

    def _retire(self, job: ChatJob) -> None:
        stream = job.stream
        retained = self.engine.finished_caches.pop(job.job_id, None)
        if retained is not None and self.checkpoints is not None:
            tokens, cache = retained
            if len(tokens) > len(job.prompt_ids):
                self.checkpoints.insert(tokens, cache, last_prompt=job.prompt_ids)
        if stream is not None and stream in self.engine.streams:
            self.engine.streams.remove(stream)
        self.completed += 1
        self._finish(job)

    @staticmethod
    def _finish(job: ChatJob) -> None:
        job.finished_at = time.perf_counter()
        job.chunks.put(None)
        job.done.set()


class ChatApp:
    """One model behind the OpenAI endpoint (``server.http.make_handler``)."""

    accepts_sampling = True
    # the request's tool list never stops prose from streaming
    streams_prose_with_tools = True

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        served_name: str,
        engine_factory: Callable[..., Any] | None = None,
        lanes: int = 1,
        max_rows: int = 16,
        max_draft: int = 32,
        pending_cap: int = 32,
        min_match: int = 4,
        default_max_tokens: int = 4096,
        context_window: int = 0,
        enable_thinking: bool = False,
        reasoning_effort: str | None = None,
        thinking_budget: int = 0,
        default_sampling: dict[str, Any] | None = None,
        max_snapshots: int = 3,
        checkpoint_slots: int | None = None,
        checkpoint_budget_bytes: int | None = 16 * 1024**3,
        model_aliases: list[str] | None = None,
        use_proposer: bool = True,
        snapshot_dir: Path | None = None,
        model_id: str = "",
    ) -> None:
        # Streamed experts (TF_EXPERT_STREAM=1) serve one sequence at a time: every forward runs on the scheduler's
        # one engine thread, and one lane keeps a second request (a client's title request, say) waiting for the
        # round boundary instead of interleaving its forward with the running one.
        if int(lanes) > 1 and getattr(model, "__dict__", {}).get("expert_stream") is not None:
            raise ValueError(f"lanes={lanes}: a model with streamed experts (TF_EXPERT_STREAM=1) serves one lane")
        # three candidate entries per conversation (history boundary, stable prefix, reply end)
        if checkpoint_slots is None:
            checkpoint_slots = max(3 * int(lanes), 8)
        self._model = model
        self.served_name = served_name
        self.model_ids = served_model_ids(served_name, model_aliases)
        self.max_batch_size = int(lanes)
        self.tokenizer = tokenizer
        self.tokenizer_lock = threading.Lock()
        self.default_max_tokens = int(default_max_tokens)
        # prompt + reply tokens a request may use (0: no limit)
        self.context_window = int(context_window)
        self.enable_thinking = bool(enable_thinking)
        self.reasoning_effort = reasoning_effort
        # the default thinking budget (0: none; a request's "thinking_budget" overrides it)
        self.thinking_budget = int(thinking_budget)
        self._think_tokens: tuple[tuple[int, ...], int] | None = None
        # ``exact_sampling.Sampling`` fields used when a request names none (None: greedy)
        self.default_sampling = dict(default_sampling) if default_sampling else None
        self.max_snapshots = int(max_snapshots)
        factory = engine_factory or LaneEngine
        self.exact_mode = {
            "mode": "exact",
            "engine": "lanes",
            "note": "every token is the model's own sample at its position; drafts only change speed",
        }
        self.stop_ids = eos_ids_of(tokenizer)
        self.engine = factory(model, max_rows=int(max_rows), max_draft=int(max_draft), pending_cap=int(pending_cap),
                              retain_finished_caches=int(checkpoint_slots) > 0)
        # Off unless asked for (a deep-prompt setup sets both): TF_PREFIX_ANCHORS=1 takes divergence anchors,
        # TF_PREFIX_SHARE=1 lets nested entries share KV arrays (model families with their own caches only)
        anchors = _env_flag("TF_PREFIX_ANCHORS")
        share = _env_flag("TF_PREFIX_SHARE") and bool(getattr(model, "lane_family", False))
        keep_tail = _env_flag("TF_PREFIX_KEEP_TAIL", True)
        self.checkpoints = (
            CheckpointStore(int(checkpoint_slots), copier=self.engine.copy_single_cache,
                            budget_bytes=checkpoint_budget_bytes, sizer=self.engine.cache_nbytes,
                            pinned_slots=max(3, self.max_snapshots), share_prefixes=share, keep_tail=keep_tail)
            if int(checkpoint_slots) > 0 else None
        )
        # a model whose prompt-cache entries hold pages of a fixed pool (Flash Next with TF_KV_PAGED=1) sizes the
        # pool for this window and frees pages by evicting the store's least recently used entries
        attach = getattr(model, "attach_prompt_cache", None)
        if self.checkpoints is not None and callable(attach):
            attach(evict=self.checkpoints.evict_lru, context_window=self.context_window)
        self.requests_completed = 0
        # background (title) requests wait this long, and while a user's request is being prepared, before they
        # are submitted, so a session's first turn, sent just after, is admitted first
        self.background_grace_s = 0.15
        self._preparing = 0
        self._preparing_lock = threading.Lock()
        self.min_match = int(min_match)
        self.use_proposer = bool(use_proposer)
        # a drafter model's proposer factory (``drafters.dflash_drafter.DFlashDrafter``), set by the CLI
        self.dflash: Any = None
        self.scheduler = Scheduler(
            self.engine,
            lanes=int(lanes),
            eos_ids=self.stop_ids,
            checkpoints=self.checkpoints,
            proposer_factory=(lambda: SuffixLookupProposer(min_match=self.min_match)) if use_proposer else None,
            snapshot_dir=snapshot_dir,
            session_dir=None if snapshot_dir is None else Path(snapshot_dir).parent / "session-snapshots",
            model_id=model_id,
            anchors=anchors,
        )
        if anchors or share:
            compact = os.environ.get("TF_CACHE_COMPACT", "").strip() not in ("", "0")
            print(f"[tensorfold] prompt cache: divergence anchors {'on' if anchors else 'off'}, "
                  f"shared prefix KV {'on' if share else 'off'}{', compact copies on' if compact else ''}", flush=True)
        if anchors:
            order = "last" if keep_tail else "in LRU order"
            print(f"[tensorfold] prompt cache: newest tail checkpoint evicted {order} "
                  f"(TF_PREFIX_KEEP_TAIL={int(keep_tail)})", flush=True)
        loaded_count = 0
        if snapshot_dir is not None and self.checkpoints is not None:
            from tensorfold.engine.prefix_snapshots import load_snapshots

            loaded_at = time.perf_counter()
            loaded = list(load_snapshots(snapshot_dir, model_id, limit=self.max_snapshots))
            for tokens, cache in reversed(loaded):        # the newest ends up most recently used
                self.checkpoints.insert(tokens, cache, last_prompt=tokens, pinned=True)
                print(f"[tensorfold] loaded system-block snapshot tokens={len(tokens)} "
                      f"in {time.perf_counter() - loaded_at:.1f}s", flush=True)
            loaded_count = len(loaded)
            del loaded
        self.scheduler.start()
        if snapshot_dir is not None and self.checkpoints is not None and not loaded_count:
            # only when these kernels have no block yet: a warmed block is pinned after the loaded ones
            self._warm_known_blocks(snapshot_dir, model_id)

    # -- request path -----------------------------------------------------------
    def render(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        thinking: bool | None = None,
    ) -> tuple[list[int], int]:
        """Prompt ids plus the length of the rendered history that prefixes them."""

        thinking = self.enable_thinking if thinking is None else bool(thinking)
        with self.tokenizer_lock:
            prompt = render_prompt_ids(self.tokenizer, messages, tools=tools, enable_thinking=thinking,
                                       reasoning_effort=self.reasoning_effort)
            history = render_prompt_ids(self.tokenizer, messages, tools=tools, enable_thinking=thinking,
                                        reasoning_effort=self.reasoning_effort, add_generation_prompt=False)
        history_len = len(history) if 0 < len(history) < len(prompt) and prompt[: len(history)] == history else 0
        return prompt, history_len

    def system_prefix_len(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
        prompt_ids: list[int], thinking: bool | None = None,
    ) -> int:
        """Tokens of the prompt that any session with this system block shares. Rendered by swapping the first user
        message for a probe and taking the common prefix, so it needs no knowledge of the chat template. Zero when
        the shared part is too short to be worth a snapshot."""

        first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
        if first_user is None:
            return 0
        probe = [*messages[:first_user], {"role": "user", "content": "⁣probe"}]
        try:
            with self.tokenizer_lock:
                other = render_prompt_ids(
                    self.tokenizer, probe, tools=tools,
                    enable_thinking=self.enable_thinking if thinking is None else bool(thinking),
                    reasoning_effort=self.reasoning_effort)
        except Exception:  # noqa: BLE001 - a template quirk must not fail the request
            return 0
        shared = longest_common_prefix(prompt_ids, other)
        return shared if shared >= 512 else 0

    def _warm_known_blocks(self, snapshot_dir: Path, model_id: str) -> None:
        """Compute, in the background, the newest system block saved by other kernels (a snapshot's bits depend
        on the kernels that computed it), so the next session finds it ready after a kernel or MLX change."""

        from tensorfold.engine.prefix_snapshots import blocks_to_warm

        blocks = blocks_to_warm(snapshot_dir, model_id)[:1]
        if not blocks:
            return
        pad = int(self.tokenizer.encode("\n", add_special_tokens=False)[-1])

        def warm() -> None:
            for tokens in blocks:
                n = len(tokens)
                job = ChatJob(
                    job_id=f"warm-{uuid.uuid4().hex[:8]}", prompt_ids=[*tokens, pad], max_tokens=1,
                    temperature=0.0, history_len=n,
                    shared_prefix_lens=tuple(m for m in (n - 2048, n - 512, n) if m >= 512), drafts=False)
                started = time.perf_counter()
                self.scheduler.submit(job)
                while job.chunks.get() is not None:
                    pass
                print(f"[tensorfold] warmed system block tokens={n} in {time.perf_counter() - started:.1f}s",
                      flush=True)

        print(f"[tensorfold] warming {len(blocks)} saved system block(s) for these kernels", flush=True)
        threading.Thread(target=warm, name="warm-blocks", daemon=True).start()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        on_delta: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _REQUEST.sampling = sampling   # per HTTP thread
        received_at = time.perf_counter()
        fields = sampling or {}
        limit = max(1, int(max_tokens or self.default_max_tokens))
        background = is_title_request(messages, tools) or fields.get("priority") == "background"
        preparing = None if background else self._Preparing(self)
        try:
            return self._chat_prepared(messages, max_tokens=limit, temperature=temperature, on_delta=on_delta,
                                       tools=tools, received_at=received_at, background=background,
                                       preparing=preparing)
        finally:
            if preparing is not None:
                preparing.release()

    class _Preparing:
        """A user's request between arrival and submission: background requests wait for these."""

        def __init__(self, app: "ChatApp") -> None:
            self.app = app
            with app._preparing_lock:
                app._preparing += 1
            self.released = False

        def release(self) -> None:
            with self.app._preparing_lock:
                if not self.released:
                    self.released = True
                    self.app._preparing -= 1

    def _chat_prepared(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        on_delta: Any | None,
        tools: list[dict[str, Any]] | None,
        received_at: float,
        background: bool,
        preparing: "ChatApp._Preparing | None",
    ) -> dict[str, Any]:
        # a request's chat_template_kwargs.enable_thinking (via the sampling fields) overrides the server's
        fields = getattr(_REQUEST, "sampling", None) or {}
        requested = fields.get("enable_thinking")
        thinking = self.enable_thinking if requested is None else bool(requested)
        prompt_ids, history_len = self.render(messages, tools, thinking=thinking)
        if not prompt_ids:
            raise RequestError("rendered prompt is empty")
        limit = max(1, int(max_tokens))
        if self.context_window:
            room = self.context_window - len(prompt_ids)
            if room < 1:
                raise RequestError(f"the prompt has {len(prompt_ids)} tokens; this server's context window is "
                                   f"{self.context_window}")
            limit = min(limit, room)
        system_len = self.system_prefix_len(messages, tools, prompt_ids, thinking=thinking)
        spec = self._resolve_sampling(fields, temperature, prompt_ids)
        drafts = self.use_proposer and fields.get("draft", True) is not False

        def make_job() -> ChatJob:
            job = ChatJob(
                job_id=f"req-{uuid.uuid4().hex[:12]}",
                prompt_ids=prompt_ids,
                max_tokens=limit,
                temperature=float(temperature),
                history_len=history_len,
                # the block's end can differ between sessions (a client that names the terminal or the date in
                # it): snapshots 512 and 2,048 tokens before it keep most of it
                shared_prefix_lens=tuple(n for n in (system_len - 2048, system_len - 512, system_len)
                                         if n >= 512) if system_len else (),
                sampling=spec,
                background=background,
                drafts=drafts,
            )
            budget = int(fields.get("thinking_budget") or self.thinking_budget) if thinking else 0
            if budget > 0:
                job.think_close, job.think_end = self._think_close()
                job.think_budget = budget if job.think_end >= 0 else 0
            if drafts:
                base: Any = SuffixLookupProposer(min_match=self.min_match)
                if self.dflash is not None:
                    # a drafter block per round; a long verbatim copy when the context has one
                    base = self.dflash.proposer(copy=base, sampling=spec)
                if tools:
                    from tensorfold.engine.tool_draft import ToolCallProposer

                    base = ToolCallProposer(_LockedTokenizer(self.tokenizer, self.tokenizer_lock), tools,
                                            len(prompt_ids), fallback=base)
                job.proposer = base
            return job

        job = make_job()
        if background:
            # the session's first turn is usually a few ms behind: let it arrive and render first
            deadline = received_at + self.background_grace_s
            while time.perf_counter() < deadline or (self._preparing and time.perf_counter() < received_at + 2.0):
                time.sleep(0.005)
        self.scheduler.submit(job)
        if preparing is not None:
            preparing.release()           # submitted: a waiting background request may go now

        collected: list[int] = []
        streamed = ""
        streamed_reasoning = ""
        streaming_done = False
        first_token_at = 0.0
        # tool calls stream as OpenAI tool_call deltas while they are written
        from tensorfold.engine.tool_draft import ToolCallStreamer

        calls_stream = ToolCallStreamer(tools) if (tools and on_delta is not None) else None
        visible_text = IncrementalText(self.tokenizer, self.tokenizer_lock)
        replay: list[int] = []        # tokens a preempted job already delivered, owed again by its rerun
        preemptions = 0
        alive = getattr(CLIENT, "alive", None)
        checked = time.perf_counter()
        if on_delta is not None:
            # a streamed write that fails means the client is gone: stop the job before the error leaves this
            # handler, or the engine decodes on for nobody up to max_tokens (an agent killed at its cap mid-stream)
            write = on_delta

            def on_delta(delta: Any) -> None:
                try:
                    write(delta)
                except BaseException:
                    if not job.cancelled:
                        job.cancelled = True
                        print(f"[tensorfold] client gone (write failed): stopping {job.job_id}", flush=True)
                    self.scheduler.finish_job(job, reason="cancelled")
                    raise
        while True:
            try:
                chunk = job.chunks.get(timeout=1.0)
            except queue.Empty:
                chunk = []
            if alive is not None and time.perf_counter() - checked >= 1.0:
                checked = time.perf_counter()
                if job.cancelled or not alive():
                    if not job.cancelled:
                        job.cancelled = True
                        print(f"[tensorfold] client gone: stopping {job.job_id}", flush=True)
                    self.scheduler.finish_job(job, reason="cancelled")   # again: it may have started since
            if chunk == []:
                continue
            if chunk is None:
                if job.preempted and job.error is None and not job.cancelled:
                    # a user's request took the engine: run again from the start (same prompt and sampling, so the
                    # same tokens) and drop what was already delivered
                    preemptions += 1
                    replay = list(collected)
                    job = make_job()
                    self.scheduler.submit(job)
                    continue
                break
            if replay:
                owed = replay[:len(chunk)]
                if list(chunk[:len(owed)]) != owed:
                    print(f"[tensorfold] rerun of a preempted request diverged: {chunk[:len(owed)]} != {owed}",
                          flush=True)
                replay = replay[len(owed):]
                chunk = chunk[len(owed):]
                if not chunk:
                    continue
            if not first_token_at:
                first_token_at = time.perf_counter()
            collected.extend(chunk)
            if on_delta is None or streaming_done:
                continue
            fresh = []
            for token in chunk:
                if token in self.stop_ids:
                    streaming_done = True
                    break
                fresh.append(token)
            text = visible_text.extend(fresh)
            if len(visible_text.tokens) and visible_text._read < len(visible_text.tokens):
                continue                    # a character still split across tokens: wait for the rest
            answer = text
            if thinking:
                # the prompt opened a think block: reasoning streams as reasoning_content until </think>
                reasoning_so_far, answer = split_thinking(text, finished=False)
                piece = reasoning_so_far[len(streamed_reasoning):]
                if piece:
                    streamed_reasoning = reasoning_so_far
                    on_delta({"reasoning_content": piece})
            shown = hide_tool_calls(answer, finished=False) if calls_stream is not None else answer
            visible = streaming_visible_text(shown)
            delta = visible[len(streamed):]
            if delta:
                streamed = visible
                on_delta(delta)
            if calls_stream is not None:
                for call_delta in calls_stream.feed(answer):   # never the reasoning: a call it mentions is not made
                    on_delta(call_delta)
        if job.error is not None:
            raise job.error

        content_tokens = strip_trailing_stops(collected, set(self.stop_ids))
        with self.tokenizer_lock:
            text = self.tokenizer.decode(content_tokens)
        if thinking:
            reasoning_text, content = split_thinking(text, finished=True)
            reasoning = reasoning_text.strip() or None
        else:
            content, reasoning = parse_harmony_output(text)
        stream = job.stream
        seconds = max(0.0, job.finished_at - job.submitted_at)
        decode_seconds = max(0.0, job.finished_at - job.prefilled_at) if job.prefilled_at else 0.0
        decode_tokens = max(0, len(collected) - 1)
        reply: dict[str, Any] = {
            "content": content,
            "reasoning": reasoning,
            "tool_calls_streamed": bool(calls_stream is not None and calls_stream.streamed),
            "finish_reason": stream.finish_reason if stream is not None else "length",
            "prompt_tokens": len(prompt_ids),
            "cached_tokens": int(job.cached_tokens),
            "completion_tokens": len(collected),
            "seconds": seconds,
            "runtime": {
                "engine": self.exact_mode["engine"],
                "tokens_per_second": (decode_tokens / decode_seconds) if decode_seconds > 0 else 0.0,
                "seconds": seconds,
                "prefill_seconds": max(0.0, job.prefilled_at - job.submitted_at) if job.prefilled_at else None,
                "time_to_first_token": (first_token_at - received_at) if first_token_at else None,
                "sampling": "exact" if spec is not None else "greedy",
                "drafts": bool(job.drafts),
                # the reply's token ids, hashed: drafted and ``"draft": false`` replies must match
                "token_sha": _token_sha(collected),
                # the narrowest verify window of the reply's rounds (drafted replies: 2 or more)
                "min_rows": int(getattr(job.stream, "min_rows", 0) or 0),
            },
        }
        if stream is not None:
            speculative: dict[str, Any] = {
                "rounds": stream.rounds,
                "drafted": stream.drafted,
                "accepted": stream.accepted,
                "acceptance_rate": (stream.accepted / stream.drafted) if stream.drafted else 0.0,
                "tokens_per_round": (len(collected) / stream.rounds) if stream.rounds else 0.0,
            }
            if stream.proposer is not None and hasattr(stream.proposer, "telemetry"):
                speculative["proposer"] = stream.proposer.telemetry()
            reply["speculative"] = speculative
        self.requests_completed += 1
        store = self.checkpoints
        print(
            f"[tensorfold] done {job.job_id} prompt={len(prompt_ids)} cached={job.cached_tokens} "
            f"tokens={len(collected)} sha={_token_sha(collected)} finish={reply['finish_reason']} "
            + (f"seed={job.sampling.seed} " if getattr(job, "sampling", None) is not None else "")
            + f"tok/s={reply['runtime']['tokens_per_second']:.1f} "
            f"ttft={(first_token_at - received_at) if first_token_at else -1:.2f}s "
            f"prefill={max(0.0, job.prefilled_at - job.started_at) if job.started_at and job.prefilled_at else -1:.2f}s "
            + (f"background preemptions={preemptions} " if background else "")
            + f"rounds={stream.rounds if stream else 0} "
            f"accepted={stream.accepted if stream else 0}/{stream.drafted if stream else 0} "
            + _draft_kinds(stream)
            + self._round_profile(stream)
            + (
                f"checkpoints={len(store)} ({store.nbytes / 1024**3:.2f} GiB, "
                f"hits={store.hits} misses={store.misses} evictions={store.evictions})"
                if store is not None
                else "checkpoints=off"
            ),
            flush=True,
        )
        return reply

    def _resolve_sampling(self, fields: dict[str, Any] | None, temperature: float,
                          prompt_ids: list[int]) -> Any:
        """The request's exact sampling, or None for greedy decoding. Fields the request leaves out come from the
        server's defaults (the model's generation_config.json, or the CLI's sampling flags). A request without a
        seed gets a random one (logged with the request, so it can be replayed): repeats of one prompt are then
        independent draws, as in llama.cpp's server. TF_SEED=prompt restores the earlier default, a hash of the
        prompt, under which the same conversation always samples the same reply."""

        from tensorfold.engine.exact_sampling import Sampling, seed_for

        fields = dict(fields or {})
        base = dict(self.default_sampling or {})
        temp = float(fields["temperature"]) if fields.get("temperature") is not None else float(base.get("temperature", 0.0))
        if temp <= 0.0:
            return None   # temperature 0: greedy
        top_p = float(fields.get("top_p", base.get("top_p", 1.0)) or 1.0)
        top_k = int(fields.get("top_k", base.get("top_k", 0)) or 0)
        if fields.get("seed") is not None:
            seed = int(fields["seed"])
        elif os.environ.get("TF_SEED", "random") == "prompt":
            seed = seed_for(prompt_ids)
        else:
            seed = int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)
        return Sampling(seed=seed, temperature=temp, top_k=top_k, top_p=top_p)

    def _think_close(self) -> tuple[tuple[int, ...], int]:
        """The tokens a thinking budget writes to close the think block ("\\n", "</think>", "\\n\\n") and the
        "</think>" id (-1 when the tokenizer has no such token)."""

        if self._think_tokens is None:
            with self.tokenizer_lock:
                end = self.tokenizer.convert_tokens_to_ids("</think>")
                unk = getattr(self.tokenizer, "unk_token_id", None)
                if not isinstance(end, int) or end < 0 or end == unk:
                    self._think_tokens = ((), -1)
                else:
                    lead = self.tokenizer.encode("\n", add_special_tokens=False)
                    trail = self.tokenizer.encode("\n\n", add_special_tokens=False)
                    self._think_tokens = ((*lead, end, *trail), int(end))
        return self._think_tokens

    def _round_profile(self, stream: Any) -> str:
        """Mean ms per round of this stream's last rounds."""

        rounds = int(getattr(stream, "rounds", 0) or 0) if stream is not None else 0
        stats = list(self.scheduler.engine.round_stats[-rounds:]) if rounds else []
        if not stats or self.scheduler.active:
            return ""
        n = len(stats)
        return (f"ms/round={sum(r.total_ms for r in stats) / n:.1f} "
                f"forward={sum(r.forward_ms for r in stats) / n:.1f} "
                f"draft={sum(r.draft_ms for r in stats) / n:.1f} post={sum(r.post_ms for r in stats) / n:.1f} "
                f"rows={sum(r.width for r in stats) / n:.1f} ")

    def close(self) -> None:
        self.scheduler.on_stop = self.save_sessions    # saved by the scheduler thread, which owns the arrays
        self.scheduler.stop(timeout=120.0)

    def save_sessions(self) -> int:
        """At shutdown, the most recent conversations' checkpoints to disk (read back on demand)."""

        if self.scheduler.session_dir is None or self.checkpoints is None:
            return 0
        return save_conversations(self.checkpoints, Path(self.scheduler.session_dir), self.scheduler.model_id)
