"""Risk bands, illustrative decisions, expected loss, and PSI utilities."""

import numpy as np
import pandas as pd


RISK_LABS = ("Very Low", "Low", "Medium", "High", "Very High")


def quant_edges(values, n_bins=10):
    """Fit quantile cut points from reference values and add open tails."""
    quant = np.quantile(np.asarray(values, dtype=float), np.linspace(0, 1, n_bins + 1))
    inner = np.unique(quant[1:-1])
    return np.concatenate(([-np.inf], inner, [np.inf]))


def risk_band(prob, edges):
    """Apply saved PD thresholds to named ordered risk bands."""
    return pd.cut(
        prob, bins=edges, labels=RISK_LABS,
        right=False, include_lowest=True, ordered=True,
    )


def decision_band(prob, cut_a, cut_b):
    """Apply illustrative approve/review/reject probability thresholds."""
    vals = np.asarray(prob)
    out = np.select(
        [vals < cut_a, vals < cut_b],
        ["Approve", "Review"],
        default="Reject",
    )
    return pd.Series(out, index=prob.index, name="decision")


def psi_num(ref, cur, edges, alpha=0.5):
    """Calculate smoothed PSI for numeric reference/current samples."""
    ref_n = np.histogram(np.asarray(ref), bins=edges)[0]
    cur_n = np.histogram(np.asarray(cur), bins=edges)[0]
    k_bin = len(ref_n)
    ref_p = (ref_n + alpha) / (ref_n.sum() + alpha * k_bin)
    cur_p = (cur_n + alpha) / (cur_n.sum() + alpha * k_bin)
    return float(np.sum((ref_p - cur_p) * np.log(ref_p / cur_p)))


def psi_cat(ref, cur, labels, alpha=0.5):
    """Calculate smoothed PSI for ordered or named categories."""
    ref_s = pd.Series(ref).astype(str)
    cur_s = pd.Series(cur).astype(str)
    ref_n = np.asarray([(ref_s == str(name)).sum() for name in labels])
    cur_n = np.asarray([(cur_s == str(name)).sum() for name in labels])
    k_bin = len(labels)
    ref_p = (ref_n + alpha) / (ref_n.sum() + alpha * k_bin)
    cur_p = (cur_n + alpha) / (cur_n.sum() + alpha * k_bin)
    return float(np.sum((ref_p - cur_p) * np.log(ref_p / cur_p)))


def psi_status(value):
    """Return a conventional illustrative PSI interpretation."""
    if value < 0.10:
        return "Low shift"
    if value < 0.25:
        return "Moderate shift"
    return "High shift"


def expected_loss(prob, lgd, ead):
    """Calculate expected loss for explicitly supplied assumptions."""
    return np.asarray(prob) * np.asarray(lgd) * np.asarray(ead)
