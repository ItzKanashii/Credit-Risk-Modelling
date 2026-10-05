"""Risk-rate tables, scoring metrics, and cross-validation helpers."""

import pandas as pd


def default_rate_by_bin(data, feature, bins, target="SeriousDlqin2yrs"):
    """Summarize borrower counts and observed default rates by feature bin.

    Bins are left-closed and right-open. Empty bins remain in the result so
    that a consistent set of risk bands can be compared across samples.
    """
    grouped = pd.cut(data[feature], bins=bins, right=False, include_lowest=True)
    temp = pd.DataFrame({"risk_bin": grouped, target: data[target]})
    rates = temp.groupby("risk_bin", observed=False).agg(
        borrowers=(target, "size"),
        defaults=(target, "sum"),
        default_rate=(target, "mean"),
    ).reset_index()
    rates["default_rate_percent"] = rates["default_rate"] * 100
    rates = rates.drop(columns="default_rate").rename(columns={"risk_bin": "bin"})
    return rates.round({"default_rate_percent": 3})


def discrimination_metrics(target, pred):
    """Calculate ranking and probability-scoring metrics."""
    import numpy as np
    from scipy.stats import ks_2samp
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    target = pd.Series(target).reset_index(drop=True)
    pred = pd.Series(pred).reset_index(drop=True)
    auc = roc_auc_score(target, pred)
    ks = ks_2samp(pred[target.eq(1)], pred[target.eq(0)]).statistic
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks,
        "pr_auc": average_precision_score(target, pred),
        "brier_score": brier_score_loss(target, pred),
    }


def calibration_table(target, pred, bins=10):
    """Summarize mean predicted PD and observed rate in quantile bins."""
    target = pd.Series(target).reset_index(drop=True)
    pred = pd.Series(pred).reset_index(drop=True)
    band = pd.qcut(pred, q=bins, labels=False, duplicates="drop") + 1
    data = pd.DataFrame({
        "risk_bin": band,
        "target": target,
        "predicted_pd": pred,
    })
    table = data.groupby("risk_bin", observed=True).agg(
        borrowers=("target", "size"),
        mean_predicted_pd=("predicted_pd", "mean"),
        actual_default_rate=("target", "mean"),
    ).reset_index()
    return table


def cross_val_auc(feat, target, seed=42, n_splits=5):
    """Estimate training-only ROC-AUC using stratified logistic CV."""
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from model import fit_logit, scale_data

    folds = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(folds.split(feat, target), start=1):
        fit_feat = feat.iloc[fit_idx]
        val_feat = feat.iloc[val_idx]
        fit_y = target.iloc[fit_idx]
        val_y = target.iloc[val_idx]
        _, fit_scaled, val_scaled = scale_data(fit_feat, val_feat)
        fit_res = fit_logit(fit_scaled, fit_y)
        val_design = sm.add_constant(val_scaled, has_constant="add")
        val_pred = fit_res.predict(val_design)
        rows.append({
            "fold": fold,
            "roc_auc": roc_auc_score(val_y, val_pred),
            "fit_rows": len(fit_idx),
            "validation_rows": len(val_idx),
        })
    return pd.DataFrame(rows)
