#!/usr/bin/env python3
"""Evaluate a raw language model on the numeric-orientation stress test.

Scoring is deterministic: sum candidate-token log probabilities for each answer
candidate appended to the prompt. No sampling or answer parsing is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return None
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model(model_id: str, device: str, dtype: str, trust_remote_code: bool):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = resolve_dtype(dtype)
    kwargs = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def candidate_logprob(model, tok, prompt: str, candidate: str, device: str, max_length: int) -> float:
    # Tokenize prompt and prompt+candidate separately to identify candidate token span.
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    full_ids = tok(prompt + candidate, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    if full_ids.shape[1] <= prompt_ids.shape[1]:
        return float("-inf")
    input_ids = full_ids.to(device)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # Candidate tokens begin after prompt_ids tokens. Because labels are shifted, the first candidate token
    # is predicted at position prompt_len - 1 in token_logp.
    start = max(0, prompt_ids.shape[1] - 1)
    cand_len = full_ids.shape[1] - prompt_ids.shape[1]
    score = token_logp[0, start : start + cand_len].sum().item()
    return float(score)


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def metrics(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not sub:
            return {"n": 0}
        correct = np.asarray([r["is_correct"] for r in sub], dtype=np.float64)
        margins = np.asarray([r["correct_margin"] for r in sub], dtype=np.float64)
        trap = [r for r in sub if r.get("regime") == "trap"]
        heuristic_errors = [r for r in trap if r.get("predicted") == r.get("surface_heuristic") and not r["is_correct"]]
        return {
            "n": len(sub),
            "accuracy": float(correct.mean()),
            "correct_margin_mean": float(margins.mean()),
            "correct_margin_l20": float(np.mean(np.sort(margins)[: max(1, int(np.ceil(0.2 * len(margins))))])),
            "trap_surface_error_rate": float(len(heuristic_errors) / len(trap)) if trap else None,
        }

    out = {"overall": metrics(records), "by_family": {}, "by_regime": {}, "by_family_regime": {}}
    for fam in sorted({r["family"] for r in records}):
        out["by_family"][fam] = metrics([r for r in records if r["family"] == fam])
    for reg in sorted({r["regime"] for r in records}):
        out["by_regime"][reg] = metrics([r for r in records if r["regime"] == reg])
    for fam in sorted({r["family"] for r in records}):
        out["by_family_regime"][fam] = {}
        for reg in sorted({r["regime"] for r in records}):
            out["by_family_regime"][fam][reg] = metrics([r for r in records if r["family"] == fam and r["regime"] == reg])
    return out


def write_md(result: Dict[str, Any], out_path: Path) -> None:
    lines = ["# Raw log-prob numeric-orientation stress results", ""]
    o = result["summary"]["overall"]
    lines.append(f"Overall accuracy: **{o['accuracy']:.4f}** over {o['n']} items")
    lines.append(f"Mean correct margin: **{o['correct_margin_mean']:.4f}**")
    lines.append("")
    lines.append("## By family/regime")
    lines.append("")
    lines.append("| Family | Regime | n | Accuracy | L20 margin | Trap surface-error rate |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for fam, regs in result["summary"]["by_family_regime"].items():
        for reg, m in regs.items():
            surf = "" if m.get("trap_surface_error_rate") is None else f"{m['trap_surface_error_rate']:.4f}"
            lines.append(f"| {fam} | {reg} | {m['n']} | {m.get('accuracy', 0):.4f} | {m.get('correct_margin_l20', 0):.4f} | {surf} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default="results/raw_logprob.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    args = p.parse_args()

    device = resolve_device(args.device)
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    items = dataset["items"]
    if args.split != "all":
        items = [it for it in items if it["split"] == args.split]

    model, tok = load_model(args.model, device, args.dtype, args.trust_remote_code)
    records = []
    for i, it in enumerate(items):
        scores = [candidate_logprob(model, tok, it["prompt"], cand, device, args.max_length) for cand in it["candidates"]]
        pred_idx = int(np.argmax(scores))
        correct_idx = int(it["correct_index"])
        correct_score = scores[correct_idx]
        other_scores = [s for j, s in enumerate(scores) if j != correct_idx]
        margin = correct_score - max(other_scores)
        pred_label = it["candidates"][pred_idx].strip()
        rec = {
            "id": it["id"],
            "family": it["family"],
            "regime": it["regime"],
            "split": it["split"],
            "scores": scores,
            "predicted_index": pred_idx,
            "predicted": pred_label,
            "correct_index": correct_idx,
            "correct": it["correct"],
            "surface_heuristic": it.get("surface_heuristic"),
            "is_correct": pred_idx == correct_idx,
            "correct_margin": float(margin),
        }
        records.append(rec)
        if (i + 1) % 50 == 0:
            print(f"scored {i+1}/{len(items)}", flush=True)

    result = {
        "experiment": "numeric_orientation_stress_raw_logprob",
        "model": args.model,
        "device": device,
        "dtype": args.dtype,
        "dataset": args.dataset,
        "split": args.split,
        "n_items": len(items),
        "summary": summarize(records),
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result, out.with_suffix(".md"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
