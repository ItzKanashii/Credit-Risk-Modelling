"""Training-only binning, WoE estimation, and scorecard utilities."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from advanced_model import short_frame


SCORE_COLS = (
    "age",
    "util",
    "debt",
    "income",
    "delinq",
    "open",
    "estate",
    "dependents",
)
FIX_CUTS = {
    "delinq": (0, 1, 2, 3, 4, np.inf),
    "estate": (0, 1, 2, 3, np.inf),
    "dependents": (0, 1, 2, 3, np.inf),
}


def fit_bins(data, target, n_bins=5, alpha=0.5):
    """Fit quantile/fixed bins and smoothed WoE tables on training rows."""
    edges = {}
    woe_map = {}
    rows = []
    good_n = int(target.eq(0).sum())
    bad_n = int(target.eq(1).sum())

    for name in SCORE_COLS:
        if name in FIX_CUTS:
            cut = np.asarray(FIX_CUTS[name], dtype=float)
        else:
            quant = np.quantile(data[name], np.linspace(0, 1, n_bins + 1))
            inner = np.unique(quant[1:-1])
            cut = np.concatenate(([-np.inf], inner, [np.inf]))
        edges[name] = cut
        band = pd.cut(data[name], cut, right=False, include_lowest=True)
        part = pd.DataFrame({"bin": band, "target": target})
        tab = part.groupby("bin", observed=True, sort=True).agg(
            borrowers=("target", "size"),
            bad=("target", "sum"),
        )
        tab["good"] = tab["borrowers"] - tab["bad"]
        k_bin = len(tab)
        tab["dist_good"] = (tab["good"] + alpha) / (good_n + alpha * k_bin)
        tab["dist_bad"] = (tab["bad"] + alpha) / (bad_n + alpha * k_bin)
        tab["bad_rate"] = tab["bad"] / tab["borrowers"]
        tab["woe"] = np.log(tab["dist_good"] / tab["dist_bad"])
        tab["iv_part"] = (tab["dist_good"] - tab["dist_bad"]) * tab["woe"]
        tab = tab.reset_index()
        tab.insert(0, "variable", name)
        rows.append(tab)
        woe_map[name] = tab.set_index("bin")["woe"].to_dict()

    bin_tab = pd.concat(rows, ignore_index=True)
    iv_tab = (
        bin_tab.groupby("variable", as_index=False)["iv_part"]
        .sum()
        .rename(columns={"iv_part": "iv"})
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
    pack = {
        "features": list(SCORE_COLS),
        "edges": edges,
        "woe": woe_map,
        "alpha": float(alpha),
    }
    train_woe = to_woe(data, pack)
    return pack, train_woe, bin_tab, iv_tab


def to_woe(data, pack):
    """Apply saved bin edges and WoE values without refitting them."""
    result = pd.DataFrame(index=data.index)
    for name in pack["features"]:
        band = pd.cut(
            data[name], pack["edges"][name], right=False,
            include_lowest=True,
        )
        result[name] = band.map(pack["woe"][name]).astype(float)
    return result


def fit_woe(data, target):
    """Fit maximum-likelihood logistic regression to WoE predictors."""
    design = sm.add_constant(data, has_constant="add")
    return sm.Logit(target, design).fit(disp=False, maxiter=100)


def score_scale(base_score=600.0, base_odds=1 / 50, pdo=20.0):
    """Derive score constants for bad-to-good odds and points-to-double-odds."""
    factor = pdo / np.log(2)
    anchor = base_score + factor * np.log(base_odds)
    return {
        "base_score": float(base_score),
        "base_odds": float(base_odds),
        "pdo": float(pdo),
        "B": float(factor),
        "A": float(anchor),
        "base_pd": float(base_odds / (1 + base_odds)),
    }


def make_pack(bin_pack, fit_res, scale, seed=47):
    """Combine fitted bins, WoE coefficients, and score-scale settings."""
    pack = dict(bin_pack)
    pack.update(scale)
    pack["coef"] = fit_res.params.drop("const").to_dict()
    pack["intercept"] = float(fit_res.params["const"])
    pack["base_points"] = pack["A"] - pack["B"] * pack["intercept"]
    pack["seed"] = int(seed)
    return pack


def score_rows(feat, pack):
    """Return PD and score for engineered baseline-feature rows."""
    data = short_frame(feat)
    enc = to_woe(data, pack)
    coef = np.asarray([pack["coef"][name] for name in pack["features"]])
    log_odds = pack["intercept"] + enc.to_numpy() @ coef
    prob = 1 / (1 + np.exp(-np.clip(log_odds, -700, 700)))
    score = pack["A"] - pack["B"] * log_odds
    return pd.DataFrame(
        {"predicted_pd": prob, "score": score}, index=feat.index
    )


def point_table(bin_tab, pack):
    """Calculate additive points for each variable/bin pair."""
    table = bin_tab[["variable", "bin", "woe"]].copy()
    table["coefficient"] = table["variable"].map(pack["coef"])
    table["points"] = -pack["B"] * table["coefficient"] * table["woe"]
    return table


def score_points(feat, pack, point_tab):
    """Sum the offset and saved bin points for each borrower."""
    data = short_frame(feat)
    total = np.full(len(data), pack["base_points"], dtype=float)
    for name in pack["features"]:
        band = pd.cut(
            data[name], pack["edges"][name], right=False,
            include_lowest=True,
        )
        part = point_tab.loc[point_tab["variable"].eq(name)]
        pts = part.set_index("bin")["points"]
        total += band.map(pts).to_numpy(dtype=float)
    return pd.Series(total, index=feat.index, name="score")


def pd_from_score(score, pack):
    """Invert the score scale to return probability of default."""
    log_odds = (pack["A"] - np.asarray(score)) / pack["B"]
    prob = 1 / (1 + np.exp(-np.clip(log_odds, -700, 700)))
    return pd.Series(prob, index=score.index, name="predicted_pd")
