# SSD streaming of routed experts

`TF_EXPERT_STREAM=1` keeps only a fixed number of each layer's routed experts in memory and reads the others from the
expert GGUF (`experts_source` in `config.json`) when a token routes to them. It is off by default. Code:
`src/tensorfold/families/qwen4_exp/expert_stream.py`.

## What it changes

Without the flag, the loader maps every layer's whole expert stack for the GPU. With it:

- Each layer has a fixed number of expert slots in host buffers that Metal reads without a copy, in the layout the
  stock modules use. The file is never mapped for the GPU.
- Decode and short continuations: a routed expert that is not in a slot is read from the file into the least
  recently used slot before the layer runs. The stock expert kernels then run on the slots. Every routed expert still
  runs, with the same bytes and the same kernels as with all experts in memory. The checks on the Hugging Face card
  found the same tokens as a full-memory server with the prompt cache off.
- Prompt chunks: the chunk's routed experts are staged into a separate buffer that has the shape of the layer's full
  stack, and the stock call runs on it with unchanged ids. Prefill does not evict the slots; after a prompt's last
  chunk, the experts its last rows routed are copied into the slots.
- Reads use one file descriptor with `F_NOCACHE` and a thread pool, so expert bytes do not stay in the page cache.
- A slot or the staging buffer is overwritten only after the GPU has finished every earlier reader of it.
- One sequence is served at a time: a batch above 1 and interleaved forwards raise. The server refuses more than one
  lane with streaming on, so a second request waits for the first.
- Drafting is not served: streaming refuses `TF_FLASH_REF_DRAFT` other than `off`, `TF_FLASH_MTP_GGUF` and
  `TF_WIRED_GIB`.

`TF_TRUNK_EMBD_ROWS=1` (separate, off by default) reads token embedding rows on the CPU from a memory map of the
trunk GGUF instead of keeping the table for the GPU; its output is bit-identical.

## Settings

Read once at load:

| variable | meaning |
|---|---|
| `TF_EXPERT_STREAM=1` | turn streaming on |
| `TF_EXPERT_ARENA_SLOTS=C` or `TF_EXPERT_ARENA_GIB=g` | slots per layer (at least 16), or the largest equal count that fits g GiB |
| `TF_EXPERT_IO_THREADS` | read threads (default 8) |
| `TF_EXPERT_RING=split\|shared` | one staging buffer per projection (`split`, default) or one shared by all, holding one projection at a time |
| `TF_EXPERT_PIN=1` | try to lock the slots and staging buffers in memory (below) |
| `TF_EXPERT_RESIDENT=L,...` | layers that keep all their experts, read once at load |
| `TF_EXPERT_LAYER_SLOTS=L:C,...` | per-layer slot counts |
| `TF_EXPERT_WARM_ROWS` | prompt rows whose experts are copied into the slots after prefill (default 64, 0 off) |
| `TF_EXPERT_STATS=path`, `TF_EXPERT_MINCORE=1`, `TF_EXPERT_DEBUG=1` | diagnostics: per-call records, cold or cached read labels, buffer checks |

With `TF_EXPERT_PIN=1` the locked bytes are capped at 0.4375 of the machine's memory and at most 21 GiB. When the
buffers do not fit under the cap, or macOS refuses the lock, the server logs why and runs unlocked instead of failing
the load. Unlocked, macOS can compress or swap the slots between uses, which slows decode.

## Cost

Every routed expert that is not in a slot costs an SSD read, so decode is far slower than with all experts in memory.
Each slot above 16 holds one more expert in every layer: fewer reads, more memory. Figures measured on one Mac are on
the [Hugging Face card](https://huggingface.co/Schiltmans/Qwen3.8-Flash-Next-Coder-cut240) (`STREAM=on ./serve.sh`).
