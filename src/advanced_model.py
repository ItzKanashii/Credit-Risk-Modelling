"""Reusable nonlinear logistic terms and model comparisons for Notebook 06."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import calibration_table, discrimination_metrics
from model import fit_logit


BASE_COLS = (
    "age",
    "util",
    "debt",
    "income",
    "delinq",
    "open",
    "estate",
    "dependents",
)
TERM_COLS = ("age", "util_log", "debt_log", "income", "delinq_log")
KNOT_Q = (0.25, 0.50, 0.75)
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def short_frame(feat):
    """Return the baseline predictors under concise internal names."""
    rename = {
        "RevolvingUtilizationOfUnsecuredLines": "util",
        "DebtRatio": "debt",
        "LogMonthlyIncome": "income",
        "TotalDelinquencies": "delinq",
        "NumberOfOpenCreditLinesAndLoans": "open",
        "NumberRealEstateLoansOrLines": "estate",
        "NumberOfDependents": "dependents",
    }
    return feat.rename(columns=rename).loc[:, BASE_COLS].copy()


class RiskTerms(BaseEstimator, TransformerMixin):
    """Add training-quantile hinge terms and a motivated interaction."""

    def fit(self, data, target=None):
        vals = self._source(data)
        self.knots_ = {}
        for name, value in vals.items():
            basis = value[value > 0] if name == "delinq_log" else value
            knot = np.unique(np.quantile(basis, KNOT_Q))
            self.knots_[name] = knot[(knot > value.min()) & (knot < value.max())]
        self.util_mean_ = float(vals["util_log"].mean())
        self.delinq_mean_ = float(vals["delinq_log"].mean())
        return self

    def transform(self, data):
        vals = self._source(data)
        out = data.copy()
        out["util_log"] = vals["util_log"]
        out["debt_log"] = vals["debt_log"]
        out["del_log"] = vals["delinq_log"]
        out["util_delinq"] = (
            (vals["util_log"] - self.util_mean_)
            * (vals["delinq_log"] - self.delinq_mean_)
        )
        prefix = {
            "age": "age",
            "util_log": "util",
            "debt_log": "debt",
            "income": "inc",
            "delinq_log": "del",
        }
        for name in TERM_COLS:
            for pos, knot in enumerate(self.knots_[name], start=1):
                col = f"h_{prefix[name]}_{pos}"
                out[col] = np.maximum(vals[name] - knot, 0.0)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.transform_columns_, dtype=object)

    @staticmethod
    def _source(data):
        return {
            "age": data["age"].to_numpy(),
            "util_log": np.log1p(data["util"].to_numpy()),
            "debt_log": np.log1p(data["debt"].to_numpy()),
            "income": data["income"].to_numpy(),
            "delinq_log": np.log1p(data["delinq"].to_numpy()),
        }

    def fit_transform(self, data, target=None, **fit_params):
        self.fit(data, target)
        out = self.transform(data)
        self.transform_columns_ = list(out.columns)
        return out


def make_reg(kind, cost=0.01):
    """Build a scaled nonlinear logistic pipeline with the chosen penalty."""
    if kind == "ridge":
        logit = LogisticRegression(
            l1_ratio=0.0, C=cost, solver="lbfgs", max_iter=3000
        )
    elif kind == "lasso":
        logit = LogisticRegression(
            l1_ratio=1.0, C=cost, solver="saga", max_iter=5000,
            tol=0.001, random_state=47,
        )
    elif kind == "elastic_net":
        logit = LogisticRegression(
            l1_ratio=0.5, C=cost, solver="saga",
            max_iter=5000, tol=0.001, random_state=47,
        )
    else:
        raise ValueError("kind must be ridge, lasso, or elastic_net")
    return Pipeline([
        ("terms", RiskTerms()),
        ("scale", StandardScaler()),
        ("logit", logit),
    ])


def cal_gap(target, pred, bins=10):
    """Return size-weighted absolute calibration error across quantile bins."""
    table = calibration_table(target, pred, bins=bins)
    weight = table["borrowers"] / table["borrowers"].sum()
    error = (table["mean_predicted_pd"] - table["actual_default_rate"]).abs()
    return float((weight * error).sum())


def score_fold(target, pred):
    """Summarize discrimination, probability error, and calibration gap."""
    values = discrimination_metrics(target, pred)
    values["cal_gap"] = cal_gap(target, pred)
    return values


def cv_mle(feat, target, nonlinear=False, seed=47, n_splits=5):
    """Score unpenalized baseline or nonlinear logistic by outer folds."""
    data = short_frame(feat)
    folds = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(
        folds.split(data, target), start=1
    ):
        fit_x = data.iloc[fit_idx]
        val_x = data.iloc[val_idx]
        fit_y = target.iloc[fit_idx]
        val_y = target.iloc[val_idx]
        if nonlinear:
            terms = RiskTerms()
            fit_x = terms.fit_transform(fit_x)
            val_x = terms.transform(val_x)
        scale = StandardScaler()
        fit_x = pd.DataFrame(
            scale.fit_transform(fit_x),
            columns=fit_x.columns,
            index=fit_x.index,
        )
        val_x = pd.DataFrame(
            scale.transform(val_x),
            columns=val_x.columns,
            index=val_x.index,
        )
        result = fit_logit(fit_x, fit_y)
        val_design = sm.add_constant(val_x, has_constant="add")
        pred = result.predict(val_design)
        rows.append({
            "fold": fold,
            "n_eff": fit_x.shape[1],
            **score_fold(val_y, pred),
        })
    return pd.DataFrame(rows)


def cv_reg(feat, target, kind, seed=47, n_splits=5):
    """Tune each regularized model inside outer training folds."""
    data = short_frame(feat)
    outer = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(
        outer.split(data, target), start=1
    ):
        fit_x = data.iloc[fit_idx]
        val_x = data.iloc[val_idx]
        fit_y = target.iloc[fit_idx]
        val_y = target.iloc[val_idx]
        inner = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=seed + fold
        )
        search = GridSearchCV(
            make_reg(kind),
            {"logit__C": C_GRID},
            scoring="roc_auc",
            cv=inner,
            n_jobs=1,
            refit=True,
        )
        search.fit(fit_x, fit_y)
        pred = search.predict_proba(val_x)[:, 1]
        coef = search.best_estimator_.named_steps["logit"].coef_[0]
        rows.append({
            "fold": fold,
            "best_c": search.best_params_["logit__C"],
            "n_eff": int((np.abs(coef) > 1e-8).sum()),
            **score_fold(val_y, pred),
        })
    return pd.DataFrame(rows)


def tune_reg(feat, target, kind, seed=47, n_splits=5):
    """Tune a regularized pipeline on training data for final refitting."""
    folds = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    search = GridSearchCV(
        make_reg(kind),
        {"logit__C": C_GRID},
        scoring="roc_auc",
        cv=folds,
        n_jobs=1,
        refit=True,
    )
    search.fit(short_frame(feat), target)
    return search


def predict_model(feat, model_pack):
    """Score rows using the selected baseline or advanced model package."""
    if model_pack["kind"] == "baseline":
        from model import predict_pd

        return predict_pd(feat, model_pack["baseline_pack"])
    data = short_frame(feat)
    if model_pack["kind"] == "nonlinear_mle":
        terms = model_pack["terms"]
        scale = model_pack["scale"]
        names = model_pack["features"]
        trans = terms.transform(data)
        vals = scale.transform(trans)
        design = pd.DataFrame(vals, columns=names, index=feat.index)
        design = sm.add_constant(design, has_constant="add")
        score = design.to_numpy() @ np.asarray(model_pack["coef"])
        prob = 1 / (1 + np.exp(-np.clip(score, -700, 700)))
        return pd.Series(prob, index=feat.index, name="predicted_pd")
    prob = model_pack["pipeline"].predict_proba(data)[:, 1]
    return pd.Series(prob, index=feat.index, name="predicted_pd")
