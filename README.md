# Credit Risk Modelling Using Logistic Regression

This project fits an interpretable logistic regression to estimate the probability of serious delinquency within two years using the supplied borrower dataset. The target is `SeriousDlqin2yrs`. The model estimates Probability of Default (PD); it does not estimate Expected Loss, Loss Given Default (LGD), or Exposure at Default (EAD).

## Repository layout

```text
.
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── raw/cs-training.csv              supplied locally; not committed
│   └── processed/cleaned_data.csv       created by Notebook 02; not committed
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning_eda.ipynb
│   ├── 03_risk_analysis.ipynb
│   ├── 04_logistic_regression.ipynb
│   └── 05_model_validation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluation.py
├── models/
│   └── logistic_regression.pkl          created by Notebook 04; not committed
└── reports/figures/                     analysis and validation figures
```

The notebooks use fixed project-relative paths. Run them from the `notebooks/` directory, starting with Notebook 01. Place the raw CSV at `data/raw/cs-training.csv`. Notebook 02 creates the cleaned dataset used by the remaining analysis.

## Setup

The project environment uses Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name credit-risk-modelling --display-name "Python (Credit Risk Modelling)"
cd notebooks
jupyter lab
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. Open the notebooks in order and select the **Python (Credit Risk Modelling)** kernel.

## Current validation results

On the seed 47 held-out 30% partition, the model has ROC-AUC **0.648**, Gini **0.297**, KS **0.216**, average precision **0.121**, and Brier score **0.0618**, compared with **0.0630** for a constant prediction using the training default rate. Five-fold stratified cross-validation within the training partition gives mean ROC-AUC **0.654** (standard deviation **0.008**; fold range **0.645–0.663**).

Observed default rates range from **2.06%** in the lowest predicted-risk decile to **13.92%** in the highest, with a decline from decile 6 to 7. Mean predicted PD is **6.798%**, compared with **6.753%** observed. These results show modest discrimination and differences between predicted and observed risk across groups. They do not establish performance over time or in another borrower population.

## Reproducibility and scope

Seed 47 controls the stratified train/test split. The saved model package contains fitted coefficients and scaling values learned from the training partition. Cross-validation refits scaling and logistic regression within each fold. The holdout was not used to fit or select the model.

This educational analysis is limited to the supplied dataset. It does not support production credit decisions or claim to be a complete regulatory scorecard.

## License

This project is released under the MIT License.
