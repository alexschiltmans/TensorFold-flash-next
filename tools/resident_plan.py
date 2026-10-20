"""What resident layers (TF_EXPERT_RESIDENT) cost: per layer of an experts GGUF, E_L, the bytes one expert takes in a
slot, the arena at C slots, and the extra bytes to hold all E_L experts instead. Reads the GGUF header only (no
tensor data, no GPU arrays).

  python tools/resident_plan.py EXPERTS.gguf --slots 160 [--resident 0,47] [--layer-slots 5:120,6:120]

The totals are the arena (every layer), the split prefill ring sized for the streamed layers, and what TF_EXPERT_PIN=1
would pin against the pin cap (``pin_cap()``, this machine's), for the given resident set and overrides; then the most layers that could be made
resident (cheapest first) with the pin cap still met.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorfold.families.qwen4_exp.expert_stream import (_PROJS, PrefillRing, _per, _layer_list,
                                                         _layer_slots, arena_bytes, pin_cap, plans_from)
from tensorfold.families.qwen4_exp.gguf_experts import read_gguf_tensors

_SHORT = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


class _Header:
    """The part of ``GGUFExperts`` that ``plans_from`` reads: ``plan[(layer, proj)]`` -> the stack's tensor info."""

    def __init__(self, path: Path) -> None:
        tensors = read_gguf_tensors(path)
        layers = sorted({int(n.split(".")[1]) for n in tensors if n.startswith("blk.") and "_exps." in n})
        self.plan = {(layer, proj): tensors[f"blk.{layer}.ffn_{_SHORT[proj]}_exps.weight"]
                     for layer in layers for proj in _PROJS}


def ring_bytes(plans: list[dict]) -> int:
    """The split ring (``PrefillRing``) for these layers, without allocating it."""

    need: dict[tuple, int] = {}
    for p in plans:
        for plan in p.values():
            for name, size, shape, _ in plan.parts():
                key = PrefillRing.kind(plan, name)
                need[key] = max(need.get(key, 0), plan.experts * _per(shape) * size)
    return sum(need.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--slots", type=int, default=160, help="C, the streamed layers' slot count")
    ap.add_argument("--resident", default="", help="TF_EXPERT_RESIDENT, for the totals")
    ap.add_argument("--layer-slots", default="", help="TF_EXPERT_LAYER_SLOTS, for the totals")
    args = ap.parse_args()
    plans = plans_from(_Header(Path(args.gguf).expanduser()))
    resident, overrides = _layer_list(args.resident), _layer_slots(args.layer_slots)
    gib, mib = 2**30, 2**20
    cap = pin_cap()

    print(f"{'layer':>5} {'E_L':>4} {'MiB/expert':>10} {'arena@C MiB':>11} {'resident MiB':>12} {'extra MiB':>9}")
    extra = {}
    for p in plans:
        layer, e = p["gate_proj"].layer, p["gate_proj"].experts
        per = sum(q.served_bytes for q in p.values())
        at_c, full = arena_bytes([p], args.slots), arena_bytes([p], e)
        extra[layer] = full - at_c
        print(f"{layer:>5} {e:>4} {per / mib:>10.3f} {at_c / mib:>11.1f} {full / mib:>12.1f} {extra[layer] / mib:>9.1f}")

    def totals(res: list[int], over: dict[int, int]) -> tuple[int, int]:
        arena = sum(arena_bytes([p], p["gate_proj"].experts if p["gate_proj"].layer in res
                                else over.get(p["gate_proj"].layer, args.slots)) for p in plans)
        return arena, ring_bytes([p for p in plans if p["gate_proj"].layer not in res])

    base_arena, base_ring = totals([], {})
    arena, ring = totals(resident, overrides)
    print(f"\nC={args.slots}, {len(plans)} layers: arena {base_arena / gib:.2f} GiB + ring {base_ring / gib:.2f} GiB = "
          f"{(base_arena + base_ring) / gib:.2f} GiB; all layers resident: "
          f"{arena_bytes(plans, max(p['gate_proj'].experts for p in plans)) / gib:.2f} GiB arena")
    if resident or overrides:
        print(f"resident {resident} overrides {overrides}: arena {arena / gib:.2f} GiB + ring {ring / gib:.2f} GiB = "
              f"{(arena + ring) / gib:.2f} GiB pinned (cap {cap / gib:.2f} GiB: "
              f"{'fits' if arena + ring <= cap else 'OVER'})")
    fit: list[int] = []
    for layer in sorted(extra, key=lambda i: (extra[i], i)):
        a, r = totals(fit + [layer], {})
        if a + r > cap:
            break
        fit.append(layer)
    a, r = totals(fit, {})
    print(f"pin cap {cap / gib:.2f} GiB at C={args.slots}: at most {len(fit)} resident layers (cheapest first: "
          f"{sorted(fit)}), {(a + r) / gib:.2f} GiB pinned")


if __name__ == "__main__":
    main()
