#!/usr/bin/env python3
"""Controlled non-saturated toy experiment for perturbation-aware SAE training.

This is a standalone mechanism check. It creates synthetic hidden states where a
small task-relevant numeric direction is embedded inside larger nuisance/template
variation. A standard SAE is trained with reconstruction+sparsity. A V-reg SAE
uses the same objective plus a Gini penalty on SAE-code responses to train
perturbation pairs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-8


@dataclass
class ToyData:
    x: np.ndarray
    y: np.ndarray
    template: np.ndarray
    drug: np.ndarray
    split: np.ndarray
    pairs: Dict[str, np.ndarray]


class SAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(d, d)).astype(np.float32)
    q, r = np.linalg.qr(a)
    sign = np.sign(np.diag(r))
    q *= sign
    return q.astype(np.float32)


def split_templates(n_templates: int, train_frac: float, dev_frac: float, rng: np.random.Generator):
    ids = np.arange(n_templates)
    rng.shuffle(ids)
    n_train = max(1, int(round(n_templates * train_frac)))
    n_dev = max(1, int(round(n_templates * dev_frac)))
    train = set(ids[:n_train].tolist())
    dev = set(ids[n_train : n_train + n_dev].tolist())
    test = set(ids[n_train + n_dev :].tolist())
    if not test:
        moved = next(iter(dev))
        dev.remove(moved)
        test.add(moved)
    return train, dev, test


def generate_toy_data(cfg: Dict[str, Any], alpha: float, seed: int) -> ToyData:
    rng = np.random.default_rng(seed)
    d = int(cfg["d_in"])
    n_templates = int(cfg["n_templates"])
    n_drugs = int(cfg["n_drugs"])
    n_repeats = int(cfg["n_repeats"])

    train_tpl, dev_tpl, test_tpl = split_templates(
        n_templates,
        float(cfg["train_template_frac"]),
        float(cfg["dev_template_frac"]),
        rng,
    )

    template_vecs = rng.normal(size=(n_templates, d)).astype(np.float32)
    drug_vecs = rng.normal(size=(n_drugs, d)).astype(np.float32)
    numeric_base = rng.normal(size=d).astype(np.float32)
    numeric_base /= np.linalg.norm(numeric_base) + EPS
    tpl_numeric_jitter = rng.normal(size=(n_templates, d)).astype(np.float32)
    tpl_numeric_jitter /= np.linalg.norm(tpl_numeric_jitter, axis=1, keepdims=True) + EPS

    r = random_orthogonal(d, rng)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    templates: List[int] = []
    drugs: List[int] = []
    splits: List[str] = []
    index_by_key: Dict[Tuple[int, int, int, int], int] = {}

    for tid in range(n_templates):
        if tid in train_tpl:
            split = "train"
        elif tid in dev_tpl:
            split = "dev"
        else:
            split = "test"
        numeric_dir = numeric_base + float(cfg["orientation_jitter"]) * tpl_numeric_jitter[tid]
        numeric_dir = numeric_dir / (np.linalg.norm(numeric_dir) + EPS)
        for did in range(n_drugs):
            for rep in range(n_repeats):
                for y in (0, 1):
                    sign = -1.0 if y == 0 else 1.0
                    h = (
                        float(cfg["template_scale"]) * template_vecs[tid]
                        + float(cfg["drug_scale"]) * drug_vecs[did]
                        + alpha * sign * numeric_dir
                        + rng.normal(scale=float(cfg["noise_std"]), size=d).astype(np.float32)
                    )
                    h = (r @ h).astype(np.float32)
                    index_by_key[(tid, did, rep, y)] = len(xs)
                    xs.append(h)
                    ys.append(y)
                    templates.append(tid)
                    drugs.append(did)
                    splits.append(split)

    def build_pairs(split_name: str) -> np.ndarray:
        pairs = []
        for tid in range(n_templates):
            if (tid in train_tpl and split_name != "train") or (tid in dev_tpl and split_name != "dev") or (tid in test_tpl and split_name != "test"):
                continue
            for did in range(n_drugs):
                for rep in range(n_repeats):
                    left = index_by_key[(tid, did, rep, 0)]
                    right = index_by_key[(tid, did, rep, 1)]
                    pairs.append((left, right))
        return np.asarray(pairs, dtype=np.int64)

    return ToyData(
        x=np.stack(xs).astype(np.float32),
        y=np.asarray(ys, dtype=np.int64),
        template=np.asarray(templates, dtype=np.int64),
        drug=np.asarray(drugs, dtype=np.int64),
        split=np.asarray(splits),
        pairs={s: build_pairs(s) for s in ("train", "dev", "test")},
    )


def gini_torch(values: torch.Tensor) -> torch.Tensor:
    values = values.flatten().clamp_min(EPS)
    if values.numel() <= 1:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    sorted_values, _ = torch.sort(values)
    n = values.numel()
    idx = torch.arange(1, n + 1, device=values.device, dtype=values.dtype)
    return (2.0 * torch.sum(idx * sorted_values) / (n * torch.sum(sorted_values))) - (n + 1.0) / n


def gini_np(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = np.maximum(v, EPS)
    if len(v) <= 1:
        return 0.0
    sv = np.sort(v)
    n = len(sv)
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * sv) / (n * np.sum(sv))) - (n + 1.0) / n)


def lower_tail_mean(values: np.ndarray, frac: float = 0.2) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    k = max(1, int(math.ceil(len(v) * frac)))
    return float(np.mean(v[:k]))


def train_sae(data: ToyData, cfg: Dict[str, Any], seed: int, *, vreg: bool) -> SAE:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.tensor(data.x[data.split == "train"], dtype=torch.float32, device=device)
    global_indices = np.where(data.split == "train")[0]
    global_to_local = {int(g): i for i, g in enumerate(global_indices)}
    train_pairs_global = data.pairs["train"]
    train_pairs_local = np.asarray(
        [(global_to_local[int(a)], global_to_local[int(b)]) for a, b in train_pairs_global], dtype=np.int64
    )
    train_pairs = torch.tensor(train_pairs_local, dtype=torch.long, device=device)

    model = SAE(int(cfg["d_in"]), int(cfg["d_sae"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-5)
    n = x_train.shape[0]
    batch_size = int(cfg["batch_size"])
    pair_batch = int(cfg["v_pair_batch"])
    steps = int(cfg["sae_steps"])
    l1_weight = float(cfg["l1_weight"])
    lambda_v = float(cfg["lambda_v"]) if vreg else 0.0

    # Dedicated generators keep init and reconstruction-batch order identical across
    # the Standard and V-reg runs (same seed), so their difference isolates the V term.
    # The pair sampler uses a separate stream so adding it does not perturb batch order.
    batch_gen = torch.Generator(device=device).manual_seed(seed + 1)
    pair_gen = torch.Generator(device=device).manual_seed(seed + 2)

    for step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device, generator=batch_gen)
        xb = x_train[idx]
        recon, z = model(xb)
        loss = F.mse_loss(recon, xb) + l1_weight * z.mean()
        if vreg and len(train_pairs) > 1:
            pidx = torch.randint(0, len(train_pairs), (min(pair_batch, len(train_pairs)),), device=device, generator=pair_gen)
            p = train_pairs[pidx]
            x_left = x_train[p[:, 0]]
            x_right = x_train[p[:, 1]]
            z_left = model.encode(x_left)
            z_right = model.encode(x_right)
            code_dist = torch.linalg.norm(z_left - z_right, dim=1)
            hidden_dist = torch.linalg.norm(x_left - x_right, dim=1).clamp_min(EPS)
            rel_response = code_dist / hidden_dist
            loss = loss + lambda_v * gini_torch(rel_response)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.cpu().eval()


@torch.no_grad()
def encode_all(model: SAE, x: np.ndarray, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    zs, recons = [], []
    for start in range(0, len(x), batch_size):
        xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32)
        recon, z = model(xb)
        zs.append(z.numpy().astype(np.float32))
        recons.append(recon.numpy().astype(np.float32))
    return np.concatenate(zs, axis=0), np.concatenate(recons, axis=0)


def fit_eval_probe(features: np.ndarray, y: np.ndarray, split: np.ndarray, c_grid: Iterable[float], seed: int) -> Dict[str, Any]:
    train = split == "train"
    dev = split == "dev"
    test = split == "test"
    best_c, best_ba = None, -1.0
    for c in c_grid:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(c), solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=seed),
        )
        clf.fit(features[train], y[train])
        pred = clf.predict(features[dev])
        ba = balanced_accuracy_score(y[dev], pred)
        if ba > best_ba:
            best_ba, best_c = ba, float(c)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(best_c), solver="lbfgs", max_iter=5000, class_weight="balanced", random_state=seed),
    )
    clf.fit(features[train], y[train])
    prob = clf.predict_proba(features[test])[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "selected_c": best_c,
        "test_auroc": float(roc_auc_score(y[test], prob)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
        "n_train": int(np.sum(train)),
        "n_dev": int(np.sum(dev)),
        "n_test": int(np.sum(test)),
    }


def pair_response_metrics(features: np.ndarray, data: ToyData, split_name: str) -> Dict[str, float]:
    pairs = data.pairs[split_name]
    left = features[pairs[:, 0]]
    right = features[pairs[:, 1]]
    dist = np.linalg.norm(left - right, axis=1)
    hdist = np.linalg.norm(data.x[pairs[:, 0]] - data.x[pairs[:, 1]], axis=1)
    rel = dist / np.maximum(hdist, EPS)
    return {
        "critical_l20_abs": lower_tail_mean(dist),
        "critical_mean_abs": float(np.mean(dist)),
        "critical_l20_rel": lower_tail_mean(rel),
        "critical_mean_rel": float(np.mean(rel)),
        "response_gini_rel": gini_np(rel),
        "response_gini_abs": gini_np(dist),
        "n_pairs": int(len(pairs)),
    }


def bootstrap_delta(values_std: np.ndarray, values_vreg: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    by_cluster = {c: np.where(clusters == c)[0] for c in unique}
    point = float(np.mean(values_vreg) - np.mean(values_std))
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_cluster[c] for c in chosen])
        boots.append(float(np.mean(values_vreg[idx]) - np.mean(values_std[idx])))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_clusters": int(len(unique)), "n_boot": int(n_boot)}


def run_one(cfg: Dict[str, Any], alpha: float, seed: int, out_dir: Path) -> Dict[str, Any]:
    data = generate_toy_data(cfg, alpha, seed)
    paired_seed = seed + 1000
    standard = train_sae(data, cfg, paired_seed, vreg=False)
    vreg = train_sae(data, cfg, paired_seed, vreg=True)
    z_std, recon_std = encode_all(standard, data.x)
    z_vreg, recon_vreg = encode_all(vreg, data.x)

    c_grid = cfg.get("probe_c_grid", [0.1, 1.0, 10.0])
    metrics: Dict[str, Any] = {
        "alpha": alpha,
        "seed": seed,
        "n_states": int(len(data.x)),
        "n_templates": int(cfg["n_templates"]),
        "splits": {s: int(np.sum(data.split == s)) for s in ("train", "dev", "test")},
        "probe": {
            "hidden": fit_eval_probe(data.x, data.y, data.split, c_grid, seed),
            "standard_code": fit_eval_probe(z_std, data.y, data.split, c_grid, seed),
            "vreg_code": fit_eval_probe(z_vreg, data.y, data.split, c_grid, seed),
        },
        "response_test": {
            "hidden": pair_response_metrics(data.x, data, "test"),
            "standard_code": pair_response_metrics(z_std, data, "test"),
            "vreg_code": pair_response_metrics(z_vreg, data, "test"),
        },
        "reconstruction_mse_train": {
            "standard": float(np.mean((recon_std[data.split == "train"] - data.x[data.split == "train"]) ** 2)),
            "vreg": float(np.mean((recon_vreg[data.split == "train"] - data.x[data.split == "train"]) ** 2)),
        },
        "reconstruction_mse_test": {
            "standard": float(np.mean((recon_std[data.split == "test"] - data.x[data.split == "test"]) ** 2)),
            "vreg": float(np.mean((recon_vreg[data.split == "test"] - data.x[data.split == "test"]) ** 2)),
        },
    }
    metrics["delta_vreg_minus_standard"] = {
        "probe_auroc": metrics["probe"]["vreg_code"]["test_auroc"] - metrics["probe"]["standard_code"]["test_auroc"],
        "probe_balanced_accuracy": metrics["probe"]["vreg_code"]["test_balanced_accuracy"] - metrics["probe"]["standard_code"]["test_balanced_accuracy"],
        "critical_l20_abs": metrics["response_test"]["vreg_code"]["critical_l20_abs"] - metrics["response_test"]["standard_code"]["critical_l20_abs"],
        "critical_l20_rel": metrics["response_test"]["vreg_code"]["critical_l20_rel"] - metrics["response_test"]["standard_code"]["critical_l20_rel"],
        "response_gini_rel": metrics["response_test"]["vreg_code"]["response_gini_rel"] - metrics["response_test"]["standard_code"]["response_gini_rel"],
    }

    # cluster bootstrap for absolute critical distances on held-out templates
    p = data.pairs["test"]
    templates = data.template[p[:, 0]]
    std_dist = np.linalg.norm(z_std[p[:, 0]] - z_std[p[:, 1]], axis=1)
    vreg_dist = np.linalg.norm(z_vreg[p[:, 0]] - z_vreg[p[:, 1]], axis=1)
    metrics["bootstrap_delta_critical_mean_abs"] = bootstrap_delta(
        std_dist,
        vreg_dist,
        templates,
        int(cfg.get("bootstrap_n", 500)),
        seed + 3000,
    )

    run_path = out_dir / "runs" / f"alpha_{alpha:.4f}_seed_{seed}.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_alpha": {}}
    for alpha in sorted({r["alpha"] for r in records}):
        rs = [r for r in records if r["alpha"] == alpha]
        def arr(path: Tuple[str, ...]) -> np.ndarray:
            vals = []
            for r in rs:
                x: Any = r
                for k in path:
                    x = x[k]
                vals.append(float(x))
            return np.asarray(vals, dtype=np.float64)
        out["by_alpha"][str(alpha)] = {
            "n_seeds": len(rs),
            "hidden_probe_auroc_mean": float(arr(("probe", "hidden", "test_auroc")).mean()),
            "standard_probe_auroc_mean": float(arr(("probe", "standard_code", "test_auroc")).mean()),
            "vreg_probe_auroc_mean": float(arr(("probe", "vreg_code", "test_auroc")).mean()),
            "delta_probe_auroc_mean": float(arr(("delta_vreg_minus_standard", "probe_auroc")).mean()),
            "delta_probe_auroc_sd": float(arr(("delta_vreg_minus_standard", "probe_auroc")).std(ddof=1)) if len(rs) > 1 else 0.0,
            "delta_critical_l20_abs_mean": float(arr(("delta_vreg_minus_standard", "critical_l20_abs")).mean()),
            "delta_critical_l20_rel_mean": float(arr(("delta_vreg_minus_standard", "critical_l20_rel")).mean()),
            "delta_response_gini_rel_mean": float(arr(("delta_vreg_minus_standard", "response_gini_rel")).mean()),
            "standard_mse_test_mean": float(arr(("reconstruction_mse_test", "standard")).mean()),
            "vreg_mse_test_mean": float(arr(("reconstruction_mse_test", "vreg")).mean()),
        }
    return out


def write_markdown(summary: Dict[str, Any], out_path: Path) -> None:
    lines = [
        "# Controlled non-saturated toy SAE results",
        "",
        "| alpha | hidden AUROC | Standard AUROC | V-reg AUROC | Δ AUROC | Δ L20 `||dz||` | Δ L20 rel D | Δ Gini rel | Std MSE | V-reg MSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha, m in summary["by_alpha"].items():
        lines.append(
            f"| {float(alpha):.4f} | {m['hidden_probe_auroc_mean']:.4f} | {m['standard_probe_auroc_mean']:.4f} | "
            f"{m['vreg_probe_auroc_mean']:.4f} | {m['delta_probe_auroc_mean']:+.4f} | "
            f"{m['delta_critical_l20_abs_mean']:+.4f} | {m['delta_critical_l20_rel_mean']:+.4f} | "
            f"{m['delta_response_gini_rel_mean']:+.4f} | {m['standard_mse_test_mean']:.4f} | {m['vreg_mse_test_mean']:.4f} |"
        )
    lines += [
        "",
        "Interpretation: look for intermediate alpha values where hidden is above chance, Standard SAE is sub-ceiling, and V-reg improves held-out probe AUROC and/or lower-tail critical response.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_smoke.json")
    parser.add_argument("--out", default="results_toy")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    records = []
    for alpha in cfg["alphas"]:
        for seed in cfg["seeds"]:
            print(f"[toy] alpha={alpha}, seed={seed}", flush=True)
            records.append(run_one(cfg, float(alpha), int(seed), out_dir))
    summary = aggregate(records)
    summary["config"] = cfg
    summary["n_runs"] = len(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out_dir / "summary.md")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
