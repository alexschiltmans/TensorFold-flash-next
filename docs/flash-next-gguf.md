# What this fork adds

The fork's code changes are in commits
[`995297c`](https://github.com/alexschiltmans/TensorFold-flash-next/commit/995297cb368bfa0f0490c55fd2a2ddf4d6888d75),
[`9ee55d5`](https://github.com/alexschiltmans/TensorFold-flash-next/commit/9ee55d5268f403957ac86c8d9c3e6bd2633b6e15)
and [`7b44450`](https://github.com/alexschiltmans/TensorFold-flash-next/commit/7b44450e003c89a36e80247a0a59bbc9abac051c),
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
  copy (`TF_CKPT_SHARE`, with `TF_CACHE_COMPACT`).
- Prompt-cache entries and prefill checkpoints that share their sparse-attention KV in pages, while the request being
  served keeps contiguous buffers (`TF_KV_PAGED`, `TF_KV_POOL_PAGES`; [kv-pages.md](kv-pages.md)).
- Four other changes to the server: it stops generating for a disconnected client, no longer preempts a background
  request such as a session title once that request has produced a token, picks a random seed when a request sends
  none (`TF_SEED=prompt` restores upstream's prompt hash), and converts string tool-call arguments to their schema
  types.

Most model-side additions are off by default; `config.json` or environment variables turn them on. The Hugging Face
repo [Schiltmans/Qwen3.8-Flash-Next-Coder-cut240](https://huggingface.co/Schiltmans/Qwen3.8-Flash-Next-Coder-cut240)
turns on the ones it uses in its `config.json` and `serve.sh`.
