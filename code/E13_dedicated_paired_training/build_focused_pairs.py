#!/usr/bin/env python3
"""Build the focused negation dataset for the dedicated GPT-2 experiment.

Self-contained (no dependency on the E3 grouping manifest). Generates a
clinical-negation minimal-pair set from explicit templates x fillers, so that
templates can be held out cleanly:

  - templates are split train / dev / test (60 / 20 / 20 by template);
  - the V-loss training pairs use TRAIN templates only;
  - the downstream probe is evaluated on TEST templates the V-regulariser
    never saw, which is what makes the downstream gain non-circular.

Outputs (into data/):
  - negation_vtrain_pairs.json : joint16-format pairs file for the V-loss
    (TRAIN templates only), family = "negation".
  - negation_probe.json        : labelled affirmed/negated examples with a
    template-held-out split field, for the probe and behavioural case study.

Label rule matches E3: clause-level negation cue -> "negated", else "affirmed".
The label is a semantic state (affirmed vs negated), never original-vs-perturbed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Each template is (affirmed, negated) with a {f} finding slot. The negated
# side uses a clause-level negation cue (no / not / does not / negative).
TEMPLATES: list[tuple[str, str]] = [
    ("The chest radiograph shows {f}.",
     "The chest radiograph shows no {f}."),
    ("There is evidence of {f} on the scan.",
     "There is no evidence of {f} on the scan."),
    ("The patient has {f}.",
     "The patient does not have {f}."),
    ("Imaging confirms the presence of {f}.",
     "Imaging does not confirm the presence of {f}."),
    ("Findings are consistent with {f}.",
     "Findings are not consistent with {f}."),
    ("The report indicates {f}.",
     "The report indicates no {f}."),
    ("Examination revealed {f}.",
     "Examination revealed no {f}."),
    ("The CT scan demonstrates {f}.",
     "The CT scan does not demonstrate {f}."),
    ("Clinical signs of {f} are present.",
     "Clinical signs of {f} are not present."),
    ("The physician noted {f} during the visit.",
     "The physician noted no {f} during the visit."),
    ("Results were positive for {f}.",
     "Results were negative for {f}."),
    ("The biopsy showed {f}.",
     "The biopsy showed no {f}."),
]

FILLERS: list[str] = [
    "pneumothorax",
    "pleural effusion",
    "consolidation",
    "a rib fracture",
    "acute hemorrhage",
    "pulmonary edema",
    "a lung mass",
    "airway inflammation",
    "arterial stenosis",
    "a pericardial effusion",
    "bowel obstruction",
    "an aortic aneurysm",
]

NEG_CUES = (
    " no ", " not ", " does not ", " negative ",
)


def negation_label(text: str) -> str:
    t = f" {text.lower()} "
    return "negated" if any(cue in t for cue in NEG_CUES) else "affirmed"


def split_templates(n_templates: int, seed: int) -> dict[int, str]:
    """Assign each template id to train / dev / test (60/20/20)."""
    rng = random.Random(seed)
    ids = list(range(n_templates))
    rng.shuffle(ids)
    n_train = round(0.60 * n_templates)
    n_dev = round(0.20 * n_templates)
    assign: dict[int, str] = {}
    for i, tid in enumerate(ids):
        if i < n_train:
            assign[tid] = "train"
        elif i < n_train + n_dev:
            assign[tid] = "dev"
        else:
            assign[tid] = "test"
    return assign


def build(seed: int) -> dict:
    template_split = split_templates(len(TEMPLATES), seed)
    examples: list[dict] = []
    vtrain_pairs: list[list[str]] = []

    for tid, (aff_t, neg_t) in enumerate(TEMPLATES):
        split = template_split[tid]
        for fid, filler in enumerate(FILLERS):
            aff = aff_t.format(f=filler)
            neg = neg_t.format(f=filler)
            pair_id = f"t{tid:02d}_f{fid:02d}"
            # Sanity: label rule must agree with intended construction.
            assert negation_label(aff) == "affirmed", (aff, negation_label(aff))
            assert negation_label(neg) == "negated", (neg, negation_label(neg))
            for side, text, label in (
                ("aff", aff, "affirmed"),
                ("neg", neg, "negated"),
            ):
                examples.append({
                    "example_id": f"{pair_id}_{side}",
                    "task": "negation_state",
                    "text": text,
                    "label": label,
                    "split": split,
                    "template_id": tid,
                    "filler_id": fid,
                    "pair_id": pair_id,
                    "side": side,
                })
            if split == "train":
                vtrain_pairs.append([aff, neg])

    payload = {
        "task": "negation_state",
        "n_templates": len(TEMPLATES),
        "n_fillers": len(FILLERS),
        "seed": seed,
        "label_rule": "clause-level negation cue -> negated, else affirmed",
        "template_split": {str(k): v for k, v in sorted(template_split.items())},
        "split_counts": {
            s: sum(1 for e in examples if e["split"] == s)
            for s in ("train", "dev", "test")
        },
        "examples": examples,
    }
    return payload, vtrain_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload, vtrain_pairs = build(args.seed)

    probe_path = args.out_dir / "negation_probe.json"
    probe_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")

    pairs_file = {
        "families": {
            "negation": {
                "description": "clinical negation minimal pairs (TRAIN templates only, for V-loss)",
                "pairs": vtrain_pairs,
            }
        }
    }
    pairs_path = args.out_dir / "negation_vtrain_pairs.json"
    pairs_path.write_text(json.dumps(pairs_file, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")

    sc = payload["split_counts"]
    print(f"templates split: {payload['template_split']}")
    print(f"examples: train={sc['train']} dev={sc['dev']} test={sc['test']} "
          f"(total {len(payload['examples'])})")
    print(f"V-loss train pairs: {len(vtrain_pairs)}")
    print(f"wrote {probe_path}")
    print(f"wrote {pairs_path}")


if __name__ == "__main__":
    main()
