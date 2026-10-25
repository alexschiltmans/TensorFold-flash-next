# What this fork adds

The fork's code changes are in commits
[`995297c`](https://github.com/alexschiltmans/TensorFold-flash-next/commit/995297cb368bfa0f0490c55fd2a2ddf4d6888d75),
[`9ee55d5`](https://github.com/alexschiltmans/TensorFold-flash-next/commit/9ee55d5268f403957ac86c8d9c3e6bd2633b6e15),
[`7b44450`](https://github.com/alexschiltmans/TensorFold-flash-next/commit/7b44450e003c89a36e80247a0a59bbc9abac051c),
[`a577cf5`](https://github.com/alexschiltmans/TensorFold-flash-next/commit/a577cf5c45ef3efed098c9d44daa56ce629233e3),
[`d71abc3`](https://github.com/alexschiltmans/TensorFold-flash-next/commit/d71abc3ecc25c5173807123cf33d16e4151d17f6)
and [`c173e66`](https://github.com/alexschiltmans/TensorFold-flash-next/commit/c173e66377e9b91eb806ee892f2a64ea0baeea5a),
based on upstream commit
[`2f8e514`](https://github.com/ashhart/TensorFold/commit/2f8e514b0b7d615df7c971627ce3c0fb7e55d93a).

- Routed experts and trunk read from llama.cpp GGUF files through mlx-kquant, mostly in place as no-copy views, and
  the n-gram table read row by row on the CPU (`gguf_experts.py`, `gguf_trunk.py`, `gguf_rows.py`).
- Reading expert files cut to a subset (`experts_cut` in `config.json`), with each router masked to the kept experts.
- An MTP head read from a GGUF, optionally cut to listed experts at load (`TF_FLASH_MTP_GGUF`, `TF_FLASH_MTP_KEEP`).
- Drafting on the reference forward, which GGUF-backed checkpoints use, with upstream's copy proposer and the GGUF
  head. It is off by default (the Hugging Face repo's `serve.sh` turns it on); with `TF_FLASH_REF_DRAFT` set, a check
  at load keeps it off unless drafted windows give the same logits, bit for bit, as the same server's one-token steps.
- Less transient memory on long prompts: prefill attention in row blocks, the forward handed to the GPU in parts,
  compact prompt-cache copies (`TF_FLASH_ATTN_MB`, `TF_FLASH_INFLIGHT`, `TF_CACHE_COMPACT`).
- Sparse prefill attention: on long prompts a row reads only the key blocks its indexer picks and its unfinished last
  block, in a new Metal kernel (`TF_FLASH_SPARSE_PREFILL`).
- Extra prompt-cache checkpoints where a prompt diverges from an earlier one and near the end of long prompts, with
  nested entries sharing KV (`TF_PREFIX_ANCHORS`, `TF_PREFIX_SHARE`); the checkpoints one prefill takes share one KV
  copy (`TF_CKPT_SHARE`, with `TF_CACHE_COMPACT`), and the newest conversation's near-end checkpoint is evicted last
  (`TF_PREFIX_KEEP_TAIL`, on by default).
- Prompt-cache entries and prefill checkpoints that share their sparse-attention KV in pages, while the request being
  served keeps contiguous buffers (`TF_KV_PAGED`, `TF_KV_POOL_PAGES`; [kv-pages.md](kv-pages.md)).
- Routed experts streamed from the expert GGUF into a fixed number of slots per layer, read with `F_NOCACHE` into
  server-owned buffers that the server tries to lock (unlocked, with a logged warning, when that fails), with
  optional fully resident layers, and token embedding rows read on the CPU (`TF_EXPERT_STREAM`,
  `TF_EXPERT_ARENA_SLOTS`, `TF_EXPERT_RESIDENT`, `TF_TRUNK_EMBD_ROWS`; [expert-streaming.md](expert-streaming.md)).
- Five other changes to the server: it stops generating for a disconnected client, no longer preempts a background
  request such as a session title once that request has produced a token, picks a random seed when a request sends
  none (`TF_SEED=prompt` restores upstream's prompt hash), converts string tool-call arguments to their schema
  types, and refuses a foreign Host or Origin, an oversized body and, with an API key set, a request without the key
  (`--allowed-host`, `--allowed-origin`, `--api-key`, `--max-body-mib`; [api.md](api.md#access-checks)).

Most model-side additions are off by default; `config.json` or environment variables turn them on. The Hugging Face
repo [Schiltmans/Qwen3.8-Flash-Next-Coder-cut240](https://huggingface.co/Schiltmans/Qwen3.8-Flash-Next-Coder-cut240)
turns on the ones it uses in its `config.json` and `serve.sh`.
