"""
Per-pair sensitivity metrics for TMLR submission audits.

Formulas match SAE_scaling/v_gini_loss_v2.py (eps=1e-8).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

EPS = 1e-8


def relative_sensitivity(z_orig: np.ndarray, z_pert: np.ndarray) -> np.ndarray:
    """s_i = ||Δz|| / (||z_orig|| + eps)."""
    z_orig = np.asarray(z_orig, dtype=np.float64)
    z_pert = np.asarray(z_pert, dtype=np.float64)
    norms = np.linalg.norm(z_orig, axis=-1)
    diffs = np.linalg.norm(z_pert - z_orig, axis=-1)
    return diffs / (norms + EPS)


def absolute_code_distance(z_orig: np.ndarray, z_pert: np.ndarray) -> np.ndarray:
    """a_i = ||z_pert - z_orig|| (unnormalized)."""
    z_orig = np.asarray(z_orig, dtype=np.float64)
    z_pert = np.asarray(z_pert, dtype=np.float64)
    return np.linalg.norm(z_pert - z_orig, axis=-1)


def code_orig_norm(z_orig: np.ndarray) -> np.ndarray:
    """b_i = ||z_orig||."""
    return np.linalg.norm(np.asarray(z_orig, dtype=np.float64), axis=-1)


def decoded_perturbation_response(
    x_hat_orig: np.ndarray, x_hat_pert: np.ndarray
) -> np.ndarray:
    """r_decode_i = ||x_hat_pert - x_hat_orig|| / (||x_hat_orig|| + eps)."""
    x_hat_orig = np.asarray(x_hat_orig, dtype=np.float64)
    x_hat_pert = np.asarray(x_hat_pert, dtype=np.float64)
    norms = np.linalg.norm(x_hat_orig, axis=-1)
    diffs = np.linalg.norm(x_hat_pert - x_hat_orig, axis=-1)
    return diffs / (norms + EPS)


def input_normalised_gain(
    z_orig: np.ndarray,
    z_pert: np.ndarray,
    h_orig: np.ndarray,
    h_pert: np.ndarray,
) -> np.ndarray:
    """g_i = s_i / (||Δh||/||h|| + eps)."""
    z_rel = relative_sensitivity(z_orig, z_pert)
    h_orig = np.asarray(h_orig, dtype=np.float64)
    h_pert = np.asarray(h_pert, dtype=np.float64)
    h_rel = np.linalg.norm(h_pert - h_orig, axis=-1) / (
        np.linalg.norm(h_orig, axis=-1) + EPS
    )
    return z_rel / (h_rel + EPS)


def hidden_perturbation_fraction(h_orig: np.ndarray, h_pert: np.ndarray) -> np.ndarray:
    """||Δh|| / ||h||."""
    return np.linalg.norm(
        np.asarray(h_pert, dtype=np.float64) - np.asarray(h_orig, dtype=np.float64),
        axis=-1,
    ) / (np.linalg.norm(np.asarray(h_orig, dtype=np.float64), axis=-1) + EPS)


def nmse(h: np.ndarray, h_hat: np.ndarray) -> float:
    """Normalized MSE: ||h - h_hat||^2 / ||h||^2."""
    h = np.asarray(h, dtype=np.float64)
    h_hat = np.asarray(h_hat, dtype=np.float64)
    residual = np.sum((h - h_hat) ** 2)
    total = np.sum(h**2)
    if total < EPS:
        return float("nan")
    return float(residual / total)


def explained_variance(h: np.ndarray, h_hat: np.ndarray) -> float:
    """1 - Var(h - h_hat) / Var(h). Scalar over all elements."""
    h = np.asarray(h, dtype=np.float64).ravel()
    h_hat = np.asarray(h_hat, dtype=np.float64).ravel()
    var_h = np.var(h)
    if var_h < EPS:
        return float("nan")
    return float(1.0 - np.var(h - h_hat) / var_h)


def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity. Shape (N,)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.sum(a * b, axis=-1)
    na = np.linalg.norm(a, axis=-1) + EPS
    nb = np.linalg.norm(b, axis=-1) + EPS
    return dot / (na * nb)


def gini_coefficient(values: np.ndarray) -> float:
    s = np.sort(np.asarray(values, dtype=np.float64))
    n = len(s)
    if n == 0:
        return float("nan")
    total = s.sum()
    if total < EPS:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * (index * s).sum() / (n * total)) - (n + 1.0) / n
    return float(np.clip(g, 0.0, 1.0))


def quantile(values: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, q))


def lower_fraction_mean(values: np.ndarray, fraction: float = 0.20) -> float:
    """L_f(D): mean of lowest ceil(f*K) values."""
    arr = np.asarray(values, dtype=np.float64)
    k = len(arr)
    if k == 0:
        return float("nan")
    n_low = max(1, math.ceil(fraction * k))
    sorted_vals = np.sort(arr)
    return float(sorted_vals[:n_low].mean())


def upper_fraction_mean(values: np.ndarray, fraction: float = 0.20) -> float:
    """U_f(D): mean of highest ceil(f*K) values."""
    arr = np.asarray(values, dtype=np.float64)
    k = len(arr)
    if k == 0:
        return float("nan")
    n_high = max(1, math.ceil(fraction * k))
    sorted_vals = np.sort(arr)
    return float(sorted_vals[-n_high:].mean())


def zero_fraction(values: np.ndarray, eps: float = 0.0) -> float:
    """p0: fraction of values with |v| <= eps."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr <= eps))


def conditional_gini(values: np.ndarray, eps: float = 0.0) -> float:
    """G+: Gini on strictly positive (|v| > eps) values; NaN if none."""
    arr = np.asarray(values, dtype=np.float64)
    pos = arr[arr > eps]
    if len(pos) == 0:
        return float("nan")
    return gini_coefficient(pos)


def decompose_v_gini(values: np.ndarray, eps: float = 0.0) -> dict[str, float]:
    """
    Floor decomposition: V = p0 + (1 - p0) * G+.

    p0 counts exact (or eps) zeros; G+ is Gini on positive values only.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {
            "V_total": float("nan"),
            "p0": float("nan"),
            "G_plus": float("nan"),
            "V_reconstructed": float("nan"),
            "n_zero": 0.0,
            "n_pos": 0.0,
            "reconstruction_error": float("nan"),
        }
    p0 = zero_fraction(arr, eps=eps)
    g_plus = conditional_gini(arr, eps=eps)
    v_total = gini_coefficient(arr)
    if math.isnan(g_plus):
        v_recon = p0
    else:
        v_recon = p0 + (1.0 - p0) * g_plus
    n_zero = float(np.sum(arr <= eps))
    n_pos = float(np.sum(arr > eps))
    return {
        "V_total": float(v_total),
        "p0": float(p0),
        "G_plus": float(g_plus) if not math.isnan(g_plus) else float("nan"),
        "V_reconstructed": float(v_recon),
        "n_zero": n_zero,
        "n_pos": n_pos,
        "reconstruction_error": float(abs(v_total - v_recon)),
    }


def summarize_distribution(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}mean": float(np.mean(arr)) if len(arr) else float("nan"),
        f"{p}median": float(np.median(arr)) if len(arr) else float("nan"),
        f"{p}q10": quantile(arr, 0.10),
        f"{p}L20": lower_fraction_mean(arr, 0.20),
        f"{p}min": float(np.min(arr)) if len(arr) else float("nan"),
        f"{p}gini": gini_coefficient(arr),
        f"{p}n": float(len(arr)),
    }


def code_sparsity_stats(z: np.ndarray, threshold: float = 1e-8) -> dict[str, float]:
    """Per-sample L0 and inactive (zero) activation fraction."""
    z = np.asarray(z, dtype=np.float64)
    active = (np.abs(z) > threshold).sum(axis=-1)
    d_sae = z.shape[-1]
    inactive_per_row = (np.abs(z) <= threshold).sum(axis=-1)
    return {
        "L0_mean": float(active.mean()),
        "L0_median": float(np.median(active)),
        "code_norm_mean": float(np.linalg.norm(z, axis=-1).mean()),
        "inactive_frac_mean": float(inactive_per_row.mean() / max(d_sae, 1)),
        "density_mean": float(active.mean() / max(d_sae, 1)),
    }


def paired_lift_summary(
    s_standard: np.ndarray,
    s_vreg: np.ndarray,
    bottom_fraction: float = 0.20,
) -> dict[str, Any]:
    """Diagnostic only — selection bias on Standard bottom tail. Not primary inference."""
    s_std = np.asarray(s_standard, dtype=np.float64)
    s_vr = np.asarray(s_vreg, dtype=np.float64)
    n = len(s_std)
    n_low = max(1, math.ceil(bottom_fraction * n))
    idx = np.argsort(s_std)
    low_idx = idx[:n_low]
    delta = s_vr - s_std
    return {
        "n_pairs": n,
        "n_bottom": n_low,
        "bottom_fraction": bottom_fraction,
        "bottom_std_mean_s": float(s_std[low_idx].mean()),
        "bottom_vreg_mean_s": float(s_vr[low_idx].mean()),
        "bottom_mean_lift": float(delta[low_idx].mean()),
        "bottom_frac_improved": float((delta[low_idx] > 0).mean()),
        "all_mean_lift": float(delta.mean()),
        "all_frac_improved": float((delta > 0).mean()),
    }


def paired_delta_summary(std_sum: dict, vreg_sum: dict) -> dict[str, float]:
    """Primary family-level comparison (distribution-level, no selection bias)."""
    return {
        "delta_mean_s": float(vreg_sum["s_mean"] - std_sum["s_mean"]),
        "delta_L20_s": float(vreg_sum["s_L20"] - std_sum["s_L20"]),
        "delta_V_gini_raw": float(vreg_sum["V_gini_raw"] - std_sum["V_gini_raw"]),
        "delta_mean_g": float(vreg_sum["g_mean"] - std_sum["g_mean"]),
        "delta_L20_g": float(vreg_sum["g_L20"] - std_sum["g_L20"]),
        "delta_V_gini_gain": float(vreg_sum["V_gini_gain"] - std_sum["V_gini_gain"]),
        "delta_L20_abs_dz": float(vreg_sum["abs_dz_L20"] - std_sum["abs_dz_L20"]),
        "delta_L20_decode_resp": float(
            vreg_sum["decode_resp_L20"] - std_sum["decode_resp_L20"]
        ),
        "mse_ratio": float(
            vreg_sum["mse_pair_mean"] / (std_sum["mse_pair_mean"] + EPS)
        ),
        "nmse_std": std_sum["nmse_mean"],
        "nmse_vreg": vreg_sum["nmse_mean"],
        "code_norm_ratio": float(
            vreg_sum["code_norm_mean"] / (std_sum["code_norm_mean"] + EPS)
        ),
    }
