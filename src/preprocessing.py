"""Preprocessing rules for the credit-risk modelling project."""

import numpy as np
import pandas as pd


def clean_data(raw_data):
    """Apply the predefined cleaning rules used in Notebook 02.

    Returns the cleaned DataFrame, an action log, and observed cleaning metrics.
    MonthlyIncome is imputed from observed non-zero values after identifier,
    duplicate, and age filtering; missing dependents are removed last. The
    original row indices are retained to support raw-to-cleaned diagnostics.
    """
    required = {"SeriousDlqin2yrs", "age", "MonthlyIncome", "NumberOfDependents"}
    missing = required.difference(raw_data.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    data = raw_data.copy()
    id_col = "Unnamed: 0"
    log = [{"step": "Starting observations", "affected": 0, "unit": "rows", "remaining_rows": len(data)}]
    if id_col in data.columns:
        data = data.drop(columns=[id_col])
        log.append({"step": "Remove index identifier", "affected": 1, "unit": "column", "remaining_rows": len(data)})

    dup_count = int(data.duplicated().sum())
    data = data.drop_duplicates(keep="first").copy()
    log.append({"step": "Remove duplicate rows", "affected": dup_count, "unit": "rows", "remaining_rows": len(data)})

    age_zero = int(data["age"].eq(0).sum())
    data = data.loc[data["age"].ne(0)].copy()
    log.append({"step": "Remove age = 0 rows", "affected": age_zero, "unit": "rows", "remaining_rows": len(data)})

    income_missing = int(data["MonthlyIncome"].isna().sum())
    income_src = data.loc[data["MonthlyIncome"].notna() & data["MonthlyIncome"].ne(0), "MonthlyIncome"]
    income_mean = float(income_src.mean())
    if not np.isfinite(income_mean):
        raise ValueError("No observed non-zero incomes are available for imputation.")
    data.loc[data["MonthlyIncome"].isna(), "MonthlyIncome"] = income_mean
    log.append({"step": "Impute missing MonthlyIncome", "affected": income_missing, "unit": "cells", "remaining_rows": len(data)})

    zero_income = int(data["MonthlyIncome"].eq(0).sum())
    dep_missing = int(data["NumberOfDependents"].isna().sum())
    data = data.dropna(subset=["NumberOfDependents"]).copy()
    log.append({"step": "Remove missing dependents", "affected": dep_missing, "unit": "rows", "remaining_rows": len(data)})

    income_info = {
        "income_donor_count": len(income_src),
        "income_mean": income_mean,
        "zero_income_retained": zero_income,
    }
    return data, pd.DataFrame(log), income_info
