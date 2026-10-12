"""Flash Next's MTP head read from a llama.cpp GGUF (ggml-org's ``mtp-Qwen3.8-Flash-Next-*.gguf``), for a model whose
trunk and experts come from GGUF files too (gguf_trunk.py, gguf_experts.py).

The file holds one ``blk.L`` layer (L = the main model's layer count) with ``nextn.*`` tensors, plus its own
``token_embd`` and ``output``, which are skipped: the head uses the main model's embedding and vocabulary head.

- Routed experts (``ffn_{gate,up,down}_exps``, 512 in ggml-org's file: the base model's head, while the Coder trunk
  keeps 256) stay the file's bytes in place: ``KQuantSwitchLinear`` over a no-copy view, as gguf_experts.py.
- Every other quantized matrix (attention, shared expert, hyper-connections, ``nextn.eh_proj``) is dequantized once
  to bf16 ``nn.Linear`` weights (~85M parameters, ~170 MB): small matrices such as the 4-row inject projections are
  then plain matmuls. Drafts only change speed (every committed token is the target's sample), so the head's
  numerics need not match a reference bit for bit.
- ``nextn.eh_proj`` [in 5120, out 2560] is ``fc_embedding`` and ``fc_hidden`` side by side on the input axis:
  columns 0..2559 act on the normed embedding, 2560..5119 on a normed stream (TF_MTP_EH_ORDER=he swaps them).
- Norms are stored as gamma = 1 + w by llama.cpp's converter (here w = gamma - 1, as gguf_trunk.py).
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.families.qwen4_exp.gguf_experts import _PROJ, _switch_glu_class, read_gguf_tensors
from tensorfold.families.qwen4_exp.mtp import FlashMTP

# ggml type id -> kquant codec (None: a dense dtype)
_CODECS = {0: None, 1: None, 30: None, 2: "q4_0", 8: "q8_0", 12: "q4_k", 13: "q5_k", 14: "q6_k", 20: "iq4_nl",
           23: "iq4_xs"}


def head_layer(names: list[str]) -> int:
    found = {int(m.group(1)) for n in names if (m := re.match(r"blk\.(\d+)\.nextn\.", n))}
    if len(found) != 1:
        raise ValueError(f"expected one blk.L with nextn tensors, found {sorted(found)}")
    return found.pop()


def head_plan(b: str) -> list[tuple[str, tuple[str, ...], str]]:
    """(FlashMTP path, GGUF tensor names, transform) for every parameter but the routed experts and fc_*."""

    p = "layers.0"
    plan = [
        ("pre_fc_norm_embedding.weight", (f"{b}.nextn.enorm.weight",), "norm_minus1"),
        ("pre_fc_norm_hidden.weight", (f"{b}.nextn.hnorm.weight",), "norm_minus1"),
        ("hyper_connection_mixer.hc_norm.weight", (f"{b}.nextn.hc_head_norm.weight",), "norm_minus1"),
        ("hyper_connection_mixer.input_mix_weight_down.weight", (f"{b}.nextn.hc_head_down.weight",), "none"),
        ("hyper_connection_mixer.input_mix_weight_up.weight", (f"{b}.nextn.hc_head_up.weight",), "none"),
        (f"{p}.mlp.gate.weight", (f"{b}.ffn_gate_inp.weight",), "none"),
        (f"{p}.mlp.shared_expert_gate.weight", (f"{b}.ffn_gate_inp_shexp.weight",), "row"),
        (f"{p}.mlp.shared_expert.gate_proj.weight", (f"{b}.ffn_gate_shexp.weight",), "none"),
        (f"{p}.mlp.shared_expert.up_proj.weight", (f"{b}.ffn_up_shexp.weight",), "none"),
        (f"{p}.mlp.shared_expert.down_proj.weight", (f"{b}.ffn_down_shexp.weight",), "none"),
        (f"{p}.self_attn.q_proj.weight", (f"{b}.attn_q.weight",), "none"),
        (f"{p}.self_attn.k_proj.weight", (f"{b}.attn_k.weight",), "none"),
        (f"{p}.self_attn.v_proj.weight", (f"{b}.attn_v.weight",), "none"),
        (f"{p}.self_attn.o_proj.weight", (f"{b}.attn_output.weight",), "none"),
        (f"{p}.self_attn.q_norm.weight", (f"{b}.attn_q_norm.weight",), "norm_minus1"),
        (f"{p}.self_attn.k_norm.weight", (f"{b}.attn_k_norm.weight",), "norm_minus1"),
        (f"{p}.self_attn.indexer.index_qk_proj.weight", (f"{b}.indexer.q_proj.weight", f"{b}.indexer.k_proj.weight"),
         "concat"),
        (f"{p}.self_attn.indexer.q_layernorm.weight", (f"{b}.indexer.q_norm.weight",), "norm_minus1"),
        (f"{p}.self_attn.indexer.k_layernorm.weight", (f"{b}.indexer.k_norm.weight",), "norm_minus1"),
    ]
    for hc, g in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
        plan += [
            (f"{p}.{hc}.hc_norm.weight", (f"{b}.{g}_norm.weight",), "norm_minus1"),
            (f"{p}.{hc}.input_mix_weight_down.weight", (f"{b}.{g}_down.weight",), "none"),
            (f"{p}.{hc}.input_mix_weight_up.weight", (f"{b}.{g}_up.weight",), "none"),
            (f"{p}.{hc}.block_inject_weight.weight", (f"{b}.{g}_inject.weight",), "none"),
        ]
    return plan


def load_gguf_head(path: str | Path, cfg: Any) -> FlashMTP:
    """The head from ``path`` for the main model's ``cfg`` (its expert count read from the file)."""

    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantSwitchLinear

    path = Path(path).expanduser().resolve()
    tensors = read_gguf_tensors(path)
    layer = head_layer(list(tensors))
    b = f"blk.{layer}"
    experts = tensors[f"{b}.ffn_gate_exps.weight"].dims[2]
    head = FlashMTP(replace(cfg, num_experts=int(experts)))
    plan = head_plan(b)
    eh = f"{b}.nextn.eh_proj.weight"
    exp_names = {f"{b}.ffn_{short}_exps.weight": proj for proj, short in _PROJ.items()}
    wanted = {n for _, names, _ in plan for n in names} | {eh} | set(exp_names)
    missing = sorted(n for n in wanted if n not in tensors)
    if missing:
        raise ValueError(f"{path.name}: missing {missing[:4]}")
    arrays, codecs, _meta, _shapes = kq.load_gguf(str(path), zero_copy=True,
                                                  skip=[n for n in tensors if n not in wanted])
    codecs = dict(codecs)

    def dense(name: str) -> mx.array:
        codec = codecs.get(name)
        a = arrays[name]
        if codec is None:
            return a
        return kq.dequantize(a, arrays[name[: -len(".weight")] + ".scales"], codec, mx.bfloat16)

    weights: dict[str, mx.array] = {}
    for tf, names, how in plan:
        src = [dense(n) for n in names]
        if how == "none":
            weights[tf] = src[0].astype(mx.bfloat16) if src[0].dtype != mx.float32 or src[0].ndim > 1 else src[0]
        elif how == "row":
            weights[tf] = src[0].reshape(1, -1)
        elif how == "norm_minus1":
            weights[tf] = src[0].astype(mx.float32) - 1.0
        elif how == "concat":
            weights[tf] = mx.concatenate(src, axis=0)
        else:
            raise ValueError(how)
    full = dense(eh)                                                     # [2560, 5120] bf16
    d = full.shape[1] // 2
    first, second = full[:, :d], full[:, d:]
    if os.environ.get("TF_MTP_EH_ORDER", "eh") == "he":
        first, second = second, first
    weights["fc_embedding.weight"], weights["fc_hidden.weight"] = first, second

    # routed experts in place (KQuantSwitchLinear over the file's bytes). TF_FLASH_MTP_KEEP=file.json ({"keep": [ids]}):
    # only those head experts are kept (a copy of their rows; the file's 512-expert view is dropped), and the head's
    # router is cut to them through set_expert_cut (masked routing + router id -> storage index), as a cut trunk.
    # Drafts stay verified by the trunk, so output is unchanged; only acceptance can move.
    keep_ids = None
    if os.environ.get("TF_FLASH_MTP_KEEP"):
        import json

        keep_ids = sorted(int(i) for i in json.loads(Path(os.environ["TF_FLASH_MTP_KEEP"]).read_text())["keep"])
    switch = head.layers[0].mlp.switch_mlp
    made = {}
    for name, proj in exp_names.items():
        old = getattr(switch, proj)
        n_exp, out_dims, in_dims = old.weight.shape
        w = arrays[name]
        if keep_ids is not None:
            w = mx.take(w, mx.array(keep_ids, dtype=mx.uint32), axis=0)
            mx.eval(w)
            n_exp = len(keep_ids)
        made[proj] = KQuantSwitchLinear(n_exp, out_dims, in_dims, False, codecs[name])
        weights[f"layers.0.mlp.switch_mlp.{proj}.weight"] = w
        weights[f"layers.0.mlp.switch_mlp.{proj}.scales"] = arrays[name[: -len(".weight")] + ".scales"]
    head.layers[0].mlp.switch_mlp = _switch_glu_class()(made["gate_proj"], made["up_proj"], made["down_proj"])
    for k in list(weights):
        if weights[k].dtype == mx.float32 and weights[k].ndim > 1 and ".switch_mlp." not in k:
            weights[k] = weights[k].astype(mx.bfloat16)
    head.load_weights(list(weights.items()), strict=True)
    mx.eval(head.parameters())
    # only now: with the file views unreferenced, the lazy norm "w - 1.0" came out as plain w (the f32 views are
    # zero-copy; MLX may reuse an input buffer it holds the only reference to), which cut acceptance from ~0.7 to ~0.4
    del arrays
    if keep_ids is not None:
        from tensorfold.families.qwen4_exp.model import set_expert_cut

        set_expert_cut(head, {0: keep_ids})
    if os.environ.get("TF_FLASH_MTP_STATS"):
        _route_stats(head, Path(os.environ["TF_FLASH_MTP_STATS"]))
    head.__dict__["gguf_source"] = {"file": str(path), "layer": layer, "experts": int(experts),
                                    "kept": len(keep_ids) if keep_ids is not None else int(experts),
                                    "expert_codec": codecs[f"{b}.ffn_gate_exps.weight"]}
    return head


def _route_stats(head: FlashMTP, out: Path) -> None:
    """TF_FLASH_MTP_STATS=file.json (diagnostics, default off): count the head's routed experts (top-k selections and
    renormalized weights, by router id) over every draft step it serves, at every depth, written every 200 calls.
    Costs a host sync a step: never in timed runs."""

    import json

    import numpy as np

    moe = head.layers[0].mlp
    orig = moe.route
    n = int(moe.gate.weight.shape[0])
    state = {"calls": 0, "rows": 0, "count": np.zeros(n), "mass": np.zeros(n)}

    def route(x):
        experts, weights = orig(x)
        e = np.array(experts.reshape(-1, experts.shape[-1]))
        w = np.array(weights.reshape(-1, weights.shape[-1]).astype(mx.float32))
        np.add.at(state["count"], e.ravel(), 1)
        np.add.at(state["mass"], e.ravel(), w.ravel())
        state["calls"] += 1
        state["rows"] += e.shape[0]
        if state["calls"] % 200 == 0:
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps({"calls": state["calls"], "rows": state["rows"],
                                       "count": state["count"].tolist(), "mass": state["mass"].tolist()}))
            tmp.replace(out)
        return experts, weights

    object.__setattr__(moe, "route", route)
