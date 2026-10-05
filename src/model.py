"""Prepare data and fit a logistic-regression PD model."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_engineering import add_features


TARGET = "SeriousDlqin2yrs"
FEATURE_COLS = (
    "age",
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "LogMonthlyIncome",
    "TotalDelinquencies",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
)


def prepare_data(data):
    """Create the modelling target and focused predictor set."""
    featured = add_features(data)
    return featured[list(FEATURE_COLS)], featured[TARGET]


def split_data(feat, target, seed=42):
    """Return stratified 70/30 train and holdout partitions."""
    return train_test_split(
        feat,
        target,
        test_size=0.30,
        random_state=seed,
        stratify=target,
    )


def scale_data(train_feat, test_feat):
    """Fit scaling on training data only and apply it to both partitions."""
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_feat),
        columns=FEATURE_COLS,
        index=train_feat.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_feat),
        columns=FEATURE_COLS,
        index=test_feat.index,
    )
    return scaler, train_scaled, test_scaled


def fit_logit(train_feat, train_target):
    """Fit maximum-likelihood logistic regression on the training sample."""
    design = sm.add_constant(train_feat, has_constant="add")
    result = sm.Logit(train_target, design).fit(disp=False, maxiter=100)
    return result


def coefficient_table(result):
    """Return coefficient inference and odds ratios per standard deviation."""
    conf = result.conf_int()
    table = pd.DataFrame({
        "coefficient": result.params,
        "std_error": result.bse,
        "z_value": result.tvalues,
        "p_value": result.pvalues,
        "ci_lower": conf[0],
        "ci_upper": conf[1],
    })
    table["odds_ratio_per_sd"] = np.exp(table["coefficient"])
    table["or_ci_lower"] = np.exp(table["ci_lower"])
    table["or_ci_upper"] = np.exp(table["ci_upper"])
    return table


def predict_pd(feat, model_pack):
    """Return fitted PDs from a saved coefficient and scaling package."""
    names = model_pack["features"]
    values = feat[names].to_numpy()
    mean = np.asarray(model_pack["scale_mean"])
    scale = np.asarray(model_pack["scale_std"])
    scaled = (values - mean) / scale
    coef = model_pack["coef"]
    slopes = np.asarray([coef[name] for name in names])
    score = coef["const"] + scaled @ slopes
    pred = 1 / (1 + np.exp(-np.clip(score, -700, 700)))
    return pd.Series(pred, index=feat.index, name="predicted_pd")
