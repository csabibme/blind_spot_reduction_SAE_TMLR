#!/usr/bin/env python3
"""Build a rule-based numeric-orientation stress benchmark.

The benchmark deliberately separates easy numeric comparisons from traps where the
larger surface number is not the task-relevant quantity.

Debiased construction (A/B families):
  - Each item is built from two *arms* with a surface number and a true quantity.
  - The correct arm is assigned to position A or B at random, alternating so that
    within every (family, regime) cell the correct answer is balanced ~50/50 across
    A and B. This removes the answer-position confound (a model that always answers
    "A" scores at chance, not at the label prior) and eliminates single-class cells.
  - `regime` (easy vs trap) is a property of the arm pair (does the larger surface
    number also carry the larger true quantity?), independent of A/B placement.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


# A/B families use symmetric templates with {a_*}/{b_*} placeholders filled from the
# arm placed at position A / position B respectively.
TOTAL_DAILY_TEMPLATES = [
    "Patient A receives {a_phrase}. Patient B receives {b_phrase}. Who receives the larger total daily dose? Answer A or B only.\nAnswer:",
    "Regimen A is {a_phrase}; regimen B is {b_phrase}. Which gives more medicine per day? Answer A or B only.\nAnswer:",
    "A: {a_phrase}. B: {b_phrase}. Which is the larger daily total? Answer A or B only.\nAnswer:",
    "Compare daily exposure. A gives {a_phrase}, while B gives {b_phrase}. Answer A or B only.\nAnswer:",
    "Which regimen has the higher total dose per 24 hours: A = {a_phrase}, B = {b_phrase}? Answer A or B only.\nAnswer:",
    "Daily dose question: patient A takes {a_phrase}; patient B takes {b_phrase}. Who gets more in one day? Answer A or B only.\nAnswer:",
]

CONC_TEMPLATES = [
    "Solution A contains {a_conc:g} mg/mL and {a_ml:g} mL is given. Solution B contains {b_conc:g} mg/mL and {b_ml:g} mL is given. Which dose contains more active drug? Answer A or B only.\nAnswer:",
    "A uses {a_ml:g} mL of a {a_conc:g} mg/mL solution. B uses {b_ml:g} mL of a {b_conc:g} mg/mL solution. Which gives more active drug? Answer A or B only.\nAnswer:",
    "Compare active amount: A = {a_conc:g} mg/mL x {a_ml:g} mL; B = {b_conc:g} mg/mL x {b_ml:g} mL. Answer A or B only.\nAnswer:",
    "Which administration is larger in mg: A with concentration {a_conc:g} mg/mL and volume {a_ml:g} mL, or B with concentration {b_conc:g} mg/mL and volume {b_ml:g} mL? Answer A or B only.\nAnswer:",
    "Dose A is prepared from {a_ml:g} mL at {a_conc:g} mg/mL. Dose B is prepared from {b_ml:g} mL at {b_conc:g} mg/mL. Which contains more drug? Answer A or B only.\nAnswer:",
    "A patient may receive A: {a_conc:g} mg/mL for {a_ml:g} mL, or B: {b_conc:g} mg/mL for {b_ml:g} mL. Which is the larger drug amount? Answer A or B only.\nAnswer:",
]

UNIT_TEMPLATES = [
    "Patient A receives {a_value:g} {a_unit}. Patient B receives {b_value:g} {b_unit}. Who receives the larger dose? Answer A or B only.\nAnswer:",
    "Compare the doses: A = {a_value:g} {a_unit}; B = {b_value:g} {b_unit}. Which is larger? Answer A or B only.\nAnswer:",
    "Which dose is greater after unit conversion, A ({a_value:g} {a_unit}) or B ({b_value:g} {b_unit})? Answer A or B only.\nAnswer:",
    "A has {a_value:g} {a_unit}, while B has {b_value:g} {b_unit}. Choose the larger dose. Answer A or B only.\nAnswer:",
    "Dose comparison: option A is {a_value:g} {a_unit}; option B is {b_value:g} {b_unit}. Which is the larger amount? Answer A or B only.\nAnswer:",
    "After converting units, which patient receives more: A with {a_value:g} {a_unit}, or B with {b_value:g} {b_unit}? Answer A or B only.\nAnswer:",
]

DIRECTION_TEMPLATES = [
    "The dose was changed from {old:g} mg to {new:g} mg. Was the dose increased or decreased? Answer increased or decreased only.\nAnswer:",
    "The prescription changed to {new:g} mg from {old:g} mg. Was this an increase or a decrease? Answer increased or decreased only.\nAnswer:",
    "A previous dose of {old:g} mg was replaced by {new:g} mg. Did the dose increase or decrease? Answer increased or decreased only.\nAnswer:",
    "The new order is {new:g} mg, whereas the old order was {old:g} mg. Is this increased or decreased? Answer increased or decreased only.\nAnswer:",
    "Dose change: old = {old:g} mg, new = {new:g} mg. What is the direction? Answer increased or decreased only.\nAnswer:",
    "The patient moved from {old:g} mg to {new:g} mg. Did the dose go up or down? Answer increased or decreased only.\nAnswer:",
    "The current dose is {new:g} mg; it used to be {old:g} mg. Did the dose increase or decrease? Answer increased or decreased only.\nAnswer:",
    "Now the dose is {new:g} mg, previously it was {old:g} mg. Was this increased or decreased? Answer increased or decreased only.\nAnswer:",
]

UNIT_TO_MG = {
    "mg": 1.0,
    "milligrams": 1.0,
    "micrograms": 0.001,
    "mcg": 0.001,
    "grams": 1000.0,
    "g": 1000.0,
}

AB_FAMILIES = ("total_daily_dose", "concentration_volume", "unit_conversion")


def split_templates(template_ids: List[str], train_frac: float, dev_frac: float, rng: random.Random) -> Dict[str, str]:
    ids = list(template_ids)
    rng.shuffle(ids)
    n_train = max(1, round(len(ids) * train_frac))
    n_dev = max(1, round(len(ids) * dev_frac))
    out = {}
    for tid in ids[:n_train]:
        out[tid] = "train"
    for tid in ids[n_train : n_train + n_dev]:
        out[tid] = "dev"
    for tid in ids[n_train + n_dev :]:
        out[tid] = "test"
    if "test" not in out.values():
        dev_ids = [k for k, v in out.items() if v == "dev"]
        out[dev_ids[0]] = "test"
    return out


# ---------------------------------------------------------------------------
# Arm generators. Each returns (arms, q_larger, s_larger) where:
#   arms      = [arm0_fields, arm1_fields]
#   q_larger  = index (0/1) of the arm with the larger TRUE quantity (the correct one)
#   s_larger  = index (0/1) of the arm with the larger SURFACE number (naive heuristic)
# regime is enforced via is_trap = (q_larger != s_larger).
# ---------------------------------------------------------------------------

def _daily_phrase(arm: Dict[str, Any]) -> str:
    if arm["h"] == 24:
        return f"{arm['mg']:g} mg once daily"
    return f"{arm['mg']:g} mg every {arm['h']:g} hours"


def make_total_daily_arms(rng: random.Random, regime: str) -> Tuple[List[Dict[str, Any]], int, int]:
    mgs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
    hours = [4, 6, 8, 12, 24]
    for _ in range(10000):
        m0, h0 = rng.choice(mgs), rng.choice(hours)
        m1, h1 = rng.choice(mgs), rng.choice(hours)
        if m0 == m1:
            continue  # surface heuristic must be defined
        t0, t1 = m0 * 24 / h0, m1 * 24 / h1
        if abs(t0 - t1) < 0.5:
            continue
        q_larger = 0 if t0 > t1 else 1
        s_larger = 0 if m0 > m1 else 1
        is_trap = q_larger != s_larger
        if (regime == "trap") != is_trap:
            continue
        return [{"mg": m0, "h": h0, "total": t0}, {"mg": m1, "h": h1, "total": t1}], q_larger, s_larger
    raise RuntimeError("Could not generate total_daily item")


def make_concentration_arms(rng: random.Random, regime: str) -> Tuple[List[Dict[str, Any]], int, int]:
    concs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    mls = [1, 2, 3, 4, 5, 6, 8]
    for _ in range(10000):
        c0, v0 = rng.choice(concs), rng.choice(mls)
        c1, v1 = rng.choice(concs), rng.choice(mls)
        if c0 == c1:
            continue
        a0, a1 = c0 * v0, c1 * v1
        if abs(a0 - a1) < 1:
            continue
        q_larger = 0 if a0 > a1 else 1
        s_larger = 0 if c0 > c1 else 1
        is_trap = q_larger != s_larger
        if (regime == "trap") != is_trap:
            continue
        return [{"conc": c0, "ml": v0, "total": a0}, {"conc": c1, "ml": v1, "total": a1}], q_larger, s_larger
    raise RuntimeError("Could not generate concentration item")


def make_unit_arms(rng: random.Random, regime: str) -> Tuple[List[Dict[str, Any]], int, int]:
    choices = [
        (0.25, "mg"), (0.5, "mg"), (0.75, "mg"), (1.0, "mg"), (2.0, "mg"),
        (250, "micrograms"), (500, "micrograms"), (650, "micrograms"), (800, "micrograms"), (1200, "micrograms"),
        (0.001, "g"), (0.002, "g"), (0.003, "g"),
    ]
    for _ in range(10000):
        v0, u0 = rng.choice(choices)
        v1, u1 = rng.choice(choices)
        if v0 == v1:
            continue  # surface heuristic must be defined on raw values
        mg0, mg1 = v0 * UNIT_TO_MG[u0], v1 * UNIT_TO_MG[u1]
        if abs(mg0 - mg1) < 0.05:
            continue
        q_larger = 0 if mg0 > mg1 else 1
        s_larger = 0 if v0 > v1 else 1
        is_trap = q_larger != s_larger
        if (regime == "trap") != is_trap:
            continue
        return [{"value": v0, "unit": u0, "mg": mg0}, {"value": v1, "unit": u1, "mg": mg1}], q_larger, s_larger
    raise RuntimeError("Could not generate unit item")


AB_ARM_MAKERS = {
    "total_daily_dose": make_total_daily_arms,
    "concentration_volume": make_concentration_arms,
    "unit_conversion": make_unit_arms,
}


def _arm_template_fields(family: str, slot: str, arm: Dict[str, Any]) -> Dict[str, Any]:
    """Render one arm into template placeholder fields for slot 'a' or 'b'."""
    if family == "total_daily_dose":
        return {f"{slot}_phrase": _daily_phrase(arm)}
    if family == "concentration_volume":
        return {f"{slot}_conc": arm["conc"], f"{slot}_ml": arm["ml"]}
    if family == "unit_conversion":
        return {f"{slot}_value": arm["value"], f"{slot}_unit": arm["unit"]}
    raise ValueError(family)


def generate_ab_item(family: str, regime: str, template_idx: int, rng: random.Random, desired_correct_pos: int) -> Dict[str, Any]:
    arms, q_larger, s_larger = AB_ARM_MAKERS[family](rng, regime)
    # Place the correct (larger-quantity) arm at the desired position for label balance.
    pos_to_arm = [None, None]
    pos_to_arm[desired_correct_pos] = q_larger
    pos_to_arm[1 - desired_correct_pos] = 1 - q_larger
    correct_index = desired_correct_pos
    surface_pos = pos_to_arm.index(s_larger)
    surface_heuristic = "A" if surface_pos == 0 else "B"

    arm_a, arm_b = arms[pos_to_arm[0]], arms[pos_to_arm[1]]
    template_values: Dict[str, Any] = {}
    template_values.update(_arm_template_fields(family, "a", arm_a))
    template_values.update(_arm_template_fields(family, "b", arm_b))
    prompt = FAMILY_TEMPLATES[family][template_idx].format(**template_values)
    return {
        "family": family,
        "regime": regime,
        "template_id": f"{family}_tpl{template_idx}",
        "prompt": prompt,
        "candidates": [" A", " B"],
        "correct_index": correct_index,
        "correct": "A" if correct_index == 0 else "B",
        "surface_heuristic": surface_heuristic,
        "values": {"position_A": arm_a, "position_B": arm_b, "surface_position": surface_heuristic},
    }


def make_change_direction(rng: random.Random, regime: str, template_idx: int) -> Dict[str, Any]:
    for _ in range(10000):
        old = rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12])
        new = rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12])
        if old == new:
            continue
        correct = "increased" if new > old else "decreased"
        correct_idx = 0 if correct == "increased" else 1
        # Order-trap heuristic: "first visible number is larger => increased".
        new_first = template_idx in {1, 3, 6, 7}
        first, second = (new, old) if new_first else (old, new)
        heuristic = "increased" if first > second else "decreased"
        is_trap = heuristic != correct
        if (regime == "trap" and is_trap) or (regime == "easy" and not is_trap):
            return {
                "values": {"old": old, "new": new, "new_first": new_first},
                "correct_index": correct_idx,
                "correct": correct,
                "surface_heuristic": heuristic,
                "template_values": {"old": old, "new": new},
            }
    raise RuntimeError("Could not generate direction item")


FAMILY_TEMPLATES = {
    "total_daily_dose": TOTAL_DAILY_TEMPLATES,
    "concentration_volume": CONC_TEMPLATES,
    "unit_conversion": UNIT_TEMPLATES,
    "change_direction": DIRECTION_TEMPLATES,
}


def generate_item(family: str, regime: str, template_idx: int, rng: random.Random, desired_correct_pos: int) -> Dict[str, Any]:
    if family in AB_FAMILIES:
        return generate_ab_item(family, regime, template_idx, rng, desired_correct_pos)
    if family == "change_direction":
        base = make_change_direction(rng, regime, template_idx)
        prompt = FAMILY_TEMPLATES[family][template_idx].format(**base["template_values"])
        return {
            "family": family,
            "regime": regime,
            "template_id": f"{family}_tpl{template_idx}",
            "prompt": prompt,
            "candidates": [" increased", " decreased"],
            "correct_index": base["correct_index"],
            "correct": base["correct"],
            "surface_heuristic": base["surface_heuristic"],
            "values": base["values"],
        }
    raise ValueError(f"Unknown family: {family}")


def build(config: Dict[str, Any]) -> Dict[str, Any]:
    rng = random.Random(int(config.get("seed", 42)))
    families = config.get("families", list(FAMILY_TEMPLATES.keys()))
    per = int(config.get("items_per_family_regime", 100))
    split_by_template: Dict[str, str] = {}
    for fam in families:
        if fam == "change_direction":
            # Split easy-compatible and trap-compatible templates separately so both regimes have test coverage.
            easy_ids = [f"{fam}_tpl{i}" for i in [1, 3, 6, 7]]
            trap_ids = [f"{fam}_tpl{i}" for i in [0, 2, 4, 5]]
            split_by_template.update(
                split_templates(easy_ids, float(config.get("train_template_frac", 0.6)), float(config.get("dev_template_frac", 0.2)), rng)
            )
            split_by_template.update(
                split_templates(trap_ids, float(config.get("train_template_frac", 0.6)), float(config.get("dev_template_frac", 0.2)), rng)
            )
        else:
            ids = [f"{fam}_tpl{i}" for i in range(len(FAMILY_TEMPLATES[fam]))]
            split_by_template.update(
                split_templates(ids, float(config.get("train_template_frac", 0.6)), float(config.get("dev_template_frac", 0.2)), rng)
            )

    items: List[Dict[str, Any]] = []
    for family in families:
        n_tpl = len(FAMILY_TEMPLATES[family])
        for regime in ("easy", "trap"):
            if family == "change_direction":
                # New-first templates are easy under the order heuristic; old-first templates are traps.
                compatible = [1, 3, 6, 7] if regime == "easy" else [0, 2, 4, 5]
            else:
                compatible = list(range(n_tpl))
            template_order = list(compatible)
            rng.shuffle(template_order)
            # Alternate the correct position *per template* so the label is balanced
            # ~50/50 within every template (and therefore within each split, since the
            # split is by template). A global i%2 alternation would alias against an
            # even-length template cycle and make each template single-class.
            per_template_counter: Dict[int, int] = {}
            for i in range(per):
                template_idx = template_order[i % len(template_order)]
                desired_correct_pos = per_template_counter.get(template_idx, 0) % 2
                per_template_counter[template_idx] = per_template_counter.get(template_idx, 0) + 1
                item = generate_item(family, regime, template_idx, rng, desired_correct_pos)
                item["id"] = f"{family}_{regime}_{i:05d}"
                item["split"] = split_by_template[item["template_id"]]
                items.append(item)

    rng.shuffle(items)
    return {
        "experiment": "numeric_orientation_stress",
        "description": "Rule-based stress test where the larger surface number can be wrong (A/B-balanced).",
        "config": config,
        "n_items": len(items),
        "items": items,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_stress_smoke.json")
    p.add_argument("--out", default="data/numeric_orientation_stress.json")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dataset = build(config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Wrote {out} with {dataset['n_items']} items")


if __name__ == "__main__":
    main()
