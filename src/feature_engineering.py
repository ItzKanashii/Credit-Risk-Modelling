"""Build total-delinquency and log-income features."""

import numpy as np
import pandas as pd


DELINQUENCY_COLS = (
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
)


def add_features(data):
    """Return a copy with total delinquency count and log monthly income."""
    required = set(DELINQUENCY_COLS) | {"MonthlyIncome"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")
    if data["MonthlyIncome"].lt(0).any():
        raise ValueError("MonthlyIncome must be non-negative for log1p.")

    result = data.copy()
    result["TotalDelinquencies"] = result[list(DELINQUENCY_COLS)].sum(axis=1)
    result["LogMonthlyIncome"] = np.log1p(result["MonthlyIncome"])
    return result
