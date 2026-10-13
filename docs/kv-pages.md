# Prompt-cache KV in pages

`TF_KV_PAGED=1`, with `TF_CACHE_COMPACT=1`, changes how Flash Next's prompt cache holds the keys and values of the
model's 12 sparse-attention layers. It is off by default.

## What it changes

Without the flag, every prompt-cache entry and every prefill checkpoint holds its own contiguous copy of the keys and
values. A long conversation's prefix is then held once for every entry that contains it: the checkpoints taken while
its prompt is read, the entry its finished reply leaves, and the checkpoints of the next turn.

With the flag:

- Each sparse-attention layer has a pool of 256-position pages. A stored entry is a list of pages in that pool. Pages
  are reference counted, so entries with the same prefix share the pages that hold it.
- The request being served keeps contiguous buffers, and decode and prefill run on them as without the flag. The
  request references the full pages it was resumed from or last stored. Its next stored copy (a checkpoint, or its
  own entry once it finishes) writes new pages only for the positions after them.
- A request resumed from an entry gets contiguous buffers filled from the entry's pages.
- An entry keeps the indexer's raw keys only from 1,024 positions below its last pooled block. The indexer reads raw
  keys only to pool blocks it has not pooled yet, so a resumed request holds zeros below that point and does not read
  them. A rollback that would reach below that point fails with an error instead of reading the zeros; drafting
  rolls back a few positions at a time.
- When a pool has no free page, the server evicts prompt-cache entries, least recently used first: conversations,
  then divergence anchors. Pinned system blocks are never evicted for pages. When no other entry is left, the copy
  that needs the page is made contiguous, as without the flag.
- `--prompt-cache-gib` counts each page once, however many entries share it.
- Disk snapshots (`--snapshot-dir`) are written as contiguous caches that hold no page references, so a snapshot is
  self-contained. A request resumed from one is paged again from its next stored copy.
- `TF_CKPT_SHARE` has no effect while the flag is on: the checkpoints share pages instead.
- The pools belong to the process and are keyed by layer: serve one Flash Next model per process with the flag on.

The code is in `src/tensorfold/families/qwen4_exp/paged_kv.py`. The server and engine do not import it: a layer
cache may provide `store_copy()`, `page_refs()`, `page_bytes()` and `snapshot_copy()`, and a model
`attach_prompt_cache()`, which the server calls once its prompt cache exists.

## Settings

| variable | default | effect |
|---|---|---|
| `TF_KV_PAGED` | off | `1`: prompt-cache entries and prefill checkpoints in pages; needs `TF_CACHE_COMPACT=1` |
| `TF_KV_POOL_PAGES` | the window in pages, plus 32 | the most pages a sparse-attention layer's pool holds |

## Pool size

A page holds 256 positions of one layer's keys and values, 512 KiB (2 KV heads, 256 dimensions, 2 bytes, keys and
values), so 6 MiB across the 12 layers. The default pool is one window of pages plus 32, rounded up to a multiple of
32:

| window | pages per layer | keys and values at most |
|---|---|---|
| 32768 | 160 | 0.94 GiB |
| 65536 | 288 | 1.69 GiB |

These sizes are computed from the model's configuration. Pool buffers are allocated 32 pages at a
time as entries need them and stay allocated while the server runs.

The default pool is sized for one conversation and its checkpoints. A second long conversation needs more: with the
default pool, the older conversation's entries are evicted when the pool fills. A pinned system block keeps its pages,
so next to a long conversation some of that conversation's copies are contiguous. Each further 64K conversation needs
256 pages a layer more (1.5 GiB across the 12 layers), by the same arithmetic.

## Serving a 64K window

With the Hugging Face repo
[Schiltmans/Qwen3.8-Flash-Next-Coder-cut240](https://huggingface.co/Schiltmans/Qwen3.8-Flash-Next-Coder-cut240):

```sh
KV_PAGES=on CONTEXT=65536 ./serve.sh
```

Its startup check then asks for an estimated 38 GB of free or cached memory instead of 36 GB. With TensorFold
directly, add `TF_KV_PAGED=1 TF_CACHE_COMPACT=1` and `--context 65536` to the other settings in that repo's
`serve.sh`.

## Measured

Qwen3.8-Flash-Next Coder cut240 with its `serve.sh` settings and drafting on, on an M4 Pro with 48 GB; no browser
running, one editor open for the 64K runs (the runs with a chat app open as well are in the evidence file). In its 64K mode `serve.sh` keeps 0.5 GiB of freed MLX buffers (2 GiB
otherwise).

| measurement | runs | result |
|---|---|---|
| replies identical to flag off, 16K-token prompts, default pool | 1 pair | 16 of 16 |
| replies identical to flag off, 16K-token prompts, a 64-page pool that fills (40 entries evicted) | 1 pair | 13 of 16 |
| decode tok/s, 16K-token prompts, flag off / on | 1 pair | 30.2 / 30.6 |
| 64K window, one ~61K-token conversation, facts recovered | 3 | 4/4 each |
| highest whole-machine wired sample at 64K, one conversation, GB | 3 | 37.20, 37.24, 37.21 (median 37.21) |

A reply counts as identical when its token ids match those of the same greedy request on a server with the flag off.
The wired sample includes macOS's own ~2.4 GB; at 64K it sits about 1.4 GB below 38.65 GB, the line at which the runs'
memory guard stops a session: close other apps before a 64K session. With the 64-page pool the three differing
replies resumed from another cached entry after evictions; without the flag, which entry a request resumes from can
change its tokens too. A pool smaller than one prompt (`TF_KV_POOL_PAGES=32` at this prompt length) empties the prompt
cache, so every request starts from an empty cache; keep the default pool (one window of pages plus 32). Every run is listed in
`evidence/window-64k.md` of the model repository.
