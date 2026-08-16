"""
Bootstrap utilities for TMLR submission.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ArrayFn = Callable[[np.ndarray], float]
DeltaFn = Callable[[np.ndarray, np.ndarray], float]


def pair_bootstrap_ci(
    values: np.ndarray,
    stat_fn: ArrayFn,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    point = stat_fn(arr)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = stat_fn(arr[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot": n_boot}


def cluster_bootstrap_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    stat_fn: ArrayFn,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(cluster_ids)
    unique = np.unique(clusters)
    if len(arr) == 0 or len(unique) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}

    rng = np.random.default_rng(seed)
    cluster_to_idx = {c: np.where(clusters == c)[0] for c in unique}
    point = stat_fn(arr)

    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in chosen])
        boots[b] = stat_fn(arr[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": n_boot,
        "n_clusters": int(len(unique)),
    }


def paired_delta_bootstrap_ci(
    std_values: np.ndarray,
    vreg_values: np.ndarray,
    delta_fn: DeltaFn,
    cluster_ids: np.ndarray | None = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for paired deltas (same index resampling)."""
    a = np.asarray(std_values, dtype=np.float64)
    b = np.asarray(vreg_values, dtype=np.float64)
    n = len(a)
    rng = np.random.default_rng(seed)
    point = delta_fn(a, b)

    def pair_resample(idx: np.ndarray) -> float:
        return delta_fn(a[idx], b[idx])

    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = pair_resample(idx)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    out = {
        "pair": {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot": n_boot},
    }

    if cluster_ids is not None:
        clusters = np.asarray(cluster_ids)
        unique = np.unique(clusters)
        if 1 < len(unique) < n:
            c2i = {c: np.where(clusters == c)[0] for c in unique}
            cl_boots = np.empty(n_boot, dtype=np.float64)
            for i in range(n_boot):
                chosen = rng.choice(unique, size=len(unique), replace=True)
                idx = np.concatenate([c2i[c] for c in chosen])
                cl_boots[i] = pair_resample(idx)
            lo, hi = np.quantile(cl_boots, [alpha, 1.0 - alpha])
            out["cluster"] = {
                "point": float(point),
                "lo": float(lo),
                "hi": float(hi),
                "n_boot": n_boot,
                "n_clusters": int(len(unique)),
            }
            out["cluster_bootstrap_valid"] = True
        else:
            out["cluster_bootstrap_valid"] = False
    return out


def default_cluster_ids(n_pairs: int, family_name: str) -> np.ndarray:
    """Each pair = own cluster. Cluster bootstrap degenerates to pair bootstrap."""
    return np.arange(n_pairs, dtype=np.int64)


def cluster_valid(cluster_ids: np.ndarray, n_pairs: int) -> bool:
    """True only if there are 2+ clusters AND fewer clusters than pairs."""
    n_clusters = len(np.unique(cluster_ids))
    return 1 < n_clusters < n_pairs
