"""Cross-fitted predictions, calibration, and bootstrap helpers for Notebook 08."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier as HistGrad
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from advanced_model import RiskTerms, short_frame, predict_model as adv_pred
from model import fit_logit, scale_data, predict_pd as base_pred
from scorecard import fit_bins, fit_woe, to_woe, score_rows


MODEL_IDS = ("baseline", "advanced", "scorecard", "forest", "boost")


def make_tree(kind, seed=47):
    """Create a fixed, documented tree candidate for comparison."""
    if kind == "forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=100,
            max_features=0.8,
            n_jobs=-1,
            random_state=seed,
        )
    if kind == "boost":
        return HistGrad(
            learning_rate=0.06,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed,
        )
    raise ValueError("kind must be forest or boost")


def _mle_pred(train_x, train_y, val_x):
    fit_res = fit_logit(train_x, train_y)
    design = sm.add_constant(val_x, has_constant="add")
    return np.asarray(fit_res.predict(design))


def _adv_pred(train_x, train_y, val_x):
    terms = RiskTerms()
    train_t = terms.fit_transform(train_x)
    val_t = terms.transform(val_x)
    scale = StandardScaler()
    train_s = pd.DataFrame(
        scale.fit_transform(train_t), columns=train_t.columns, index=train_t.index
    )
    val_s = pd.DataFrame(
        scale.transform(val_t), columns=val_t.columns, index=val_t.index
    )
    return _mle_pred(train_s, train_y, val_s)


def _woe_pred(train_x, train_y, val_x):
    bin_pack, train_w, _, _ = fit_bins(train_x, train_y)
    val_w = to_woe(val_x, bin_pack)
    fit_res = fit_woe(train_w, train_y)
    design = sm.add_constant(val_w, has_constant="add")
    return np.asarray(fit_res.predict(design))


def cross_preds(feat, target, seed=47, n_splits=5):
    """Generate out-of-fold predictions with all transforms fit per fold."""
    folds = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    out = pd.DataFrame(index=feat.index, columns=MODEL_IDS, dtype=float)
    out["fold"] = -1
    raw_x = short_frame(feat)
    for fold, (fit_idx, val_idx) in enumerate(folds.split(feat, target), start=1):
        fit_feat = feat.iloc[fit_idx]
        val_feat = feat.iloc[val_idx]
        fit_y = target.iloc[fit_idx]
        fit_x = raw_x.iloc[fit_idx]
        val_x = raw_x.iloc[val_idx]

        _, base_fit, base_val = scale_data(fit_feat, val_feat)
        out.loc[val_feat.index, "baseline"] = _mle_pred(
            base_fit, fit_y, base_val
        )
        out.loc[val_feat.index, "advanced"] = _adv_pred(
            fit_x, fit_y, val_x
        )
        out.loc[val_feat.index, "scorecard"] = _woe_pred(
            fit_x, fit_y, val_x
        )
        for kind in ("forest", "boost"):
            tree = make_tree(kind, seed + fold)
            tree.fit(fit_x, fit_y)
            out.loc[val_feat.index, kind] = tree.predict_proba(val_x)[:, 1]
        out.loc[val_feat.index, "fold"] = fold
    out["fold"] = out["fold"].astype(int)
    return out


def fit_tree(kind, feat, target, seed=47):
    """Fit one fixed tree candidate on the complete training partition."""
    model = make_tree(kind, seed)
    model.fit(short_frame(feat), target)
    return model


def _logit_prob(prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(prob / (1 - prob)).reshape(-1, 1)


def fit_cal(prob, target, method):
    """Fit Platt or isotonic calibration on supplied training predictions."""
    if method == "raw":
        return {"method": "raw", "model": None}
    if method == "platt":
        model = LogisticRegression(C=1e6, max_iter=1000, random_state=47)
        model.fit(_logit_prob(prob), target)
        return {"method": method, "model": model}
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(np.asarray(prob), np.asarray(target))
        return {"method": method, "model": model}
    raise ValueError("method must be raw, platt, or isotonic")


def apply_cal(prob, cal_pack):
    """Apply a saved calibration mapping to predicted probabilities."""
    method = cal_pack["method"]
    if method == "raw":
        return np.asarray(prob, dtype=float)
    if method == "platt":
        return cal_pack["model"].predict_proba(_logit_prob(prob))[:, 1]
    return cal_pack["model"].predict(np.asarray(prob, dtype=float))


def cross_cal(prob, target, method, seed=47, n_splits=5):
    """Cross-validate a calibrator using only training-fold predictions."""
    prob = pd.Series(prob, index=target.index, dtype=float)
    if method == "raw":
        return prob.copy()
    folds = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    out = pd.Series(np.nan, index=target.index, dtype=float)
    for fit_idx, val_idx in folds.split(prob, target):
        fit_p = prob.iloc[fit_idx]
        val_p = prob.iloc[val_idx]
        fit_y = target.iloc[fit_idx]
        cal = fit_cal(fit_p, fit_y, method)
        out.iloc[val_idx] = apply_cal(val_p, cal)
    return out


def auc_ci(target, pred, n_boot=400, seed=47):
    """Return percentile AUC confidence limits from stratified bootstrap."""
    target = np.asarray(target)
    pred = np.asarray(pred)
    pos = np.flatnonzero(target == 1)
    neg = np.flatnonzero(target == 0)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate((
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ))
        vals[b] = roc_auc_score(target[idx], pred[idx])
    return tuple(np.quantile(vals, [0.025, 0.975]))


def auc_diff_ci(target, pred_a, pred_b, n_boot=400, seed=47):
    """Return paired percentile limits for the difference in AUC."""
    target = np.asarray(target)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    pos = np.flatnonzero(target == 1)
    neg = np.flatnonzero(target == 0)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = np.concatenate((
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ))
        auc_a = roc_auc_score(target[idx], pred_a[idx])
        auc_b = roc_auc_score(target[idx], pred_b[idx])
        vals[b] = auc_a - auc_b
    return tuple(np.quantile(vals, [0.025, 0.975]))


def predict_candidate(feat, model_pack):
    """Predict PD with a saved champion model and calibration mapping."""
    kind = model_pack["model_id"]
    core = model_pack["model"]
    if kind == "baseline":
        raw = base_pred(feat, core)
    elif kind == "advanced":
        raw = adv_pred(feat, core)
    elif kind == "scorecard":
        raw = score_rows(feat, core)["predicted_pd"]
    else:
        raw = core.predict_proba(short_frame(feat))[:, 1]
    prob = apply_cal(raw, model_pack["cal"])
    return pd.Series(prob, index=feat.index, name="predicted_pd")
