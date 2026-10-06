# Credit Risk Modelling

This project studies borrower-level probability of serious delinquency within two years using the `cs-training.csv` dataset. The target is `SeriousDlqin2yrs`. It compares logistic regression, a Weight of Evidence (WoE) scorecard, and tree-based models, with emphasis on probability estimation, validation, interpretation, and risk segmentation.

The models estimate Probability of Default (PD) for this target. The dataset does not contain Loss Given Default (LGD) or Exposure at Default (EAD), so the project does not estimate portfolio Expected Loss or support a complete credit decision policy.

## Project contents

```text
.
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── raw/cs-training.csv
│   └── processed/cleaned_data.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning_eda.ipynb
│   ├── 03_risk_analysis.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_model_validation.ipynb
│   ├── 06_advanced_pd_modeling.ipynb
│   ├── 07_scorecard_and_woe.ipynb
│   ├── 08_model_comparison_and_calibration.ipynb
│   └── 09_risk_decision_and_monitoring.ipynb
├── src/
│   ├── advanced_model.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── model_comparison.py
│   ├── monitoring.py
│   ├── preprocessing.py
│   └── scorecard.py
├── models/                    # Model packages saved by notebooks
└── reports/figures/           # Notebook-generated figures
```

The raw dataset is supplied locally and is not committed. Notebook 02 writes the cleaned data used by later notebooks. Generated model packages are stored under `models/`; figures are stored under `reports/figures/`.

## Setup and use

The notebooks were developed with Python 3.12. From the repository root, create and activate the environment, install the listed packages, then launch Jupyter from `notebooks/` so the relative data and source paths resolve as written.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name credit-risk-modelling --display-name "Python (Credit Risk Modelling)"
cd notebooks
jupyter lab
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. Select the **Python (Credit Risk Modelling)** kernel. Place `cs-training.csv` in `data/raw/` before running Notebook 01. Run notebooks 01 through 09 in order; Notebook 02 creates `data/processed/cleaned_data.csv` for the subsequent analysis.

## Results

The metrics below are from Notebook 08 using the seed 47 stratified split with 30% held out. The holdout contains 43,669 borrowers. Calibration methods were selected with training-only out-of-fold predictions; holdout outcomes were reserved for final evaluation.

| Model | Calibration | Holdout ROC-AUC | Gini | KS | PR-AUC | Brier score |
|---|---|---:|---:|---:|---:|---:|
| Baseline logistic regression | Isotonic | 0.6480 | 0.2961 | 0.2152 | 0.1170 | 0.0616 |
| Advanced Ridge logistic regression | Platt | 0.8528 | 0.7056 | 0.5527 | 0.3673 | 0.0512 |
| WoE scorecard | Isotonic | 0.8511 | 0.7022 | 0.5538 | 0.3468 | 0.0513 |
| Random Forest | Platt | 0.8579 | 0.7157 | 0.5653 | 0.3827 | 0.0505 |
| Gradient Boosting, selected model | Raw probabilities | 0.8601 | 0.7201 | 0.5671 | 0.3873 | 0.0503 |

Notebook 08 selected Gradient Boosting using training-only cross-validation. Its mean five-fold ROC-AUC was 0.8623, mean Brier score was 0.0499, and mean calibration gap was 0.0031. The holdout ROC-AUC was 0.8601 (95% bootstrap interval 0.8532 to 0.8658). The final comparison used raw probabilities for Gradient Boosting, Platt calibration for the advanced logistic model and Random Forest, and isotonic calibration for the baseline and scorecard. These results describe one random split, not performance over time or in another borrower population.

The WoE scorecard provides a more transparent alternative with discrimination close to the advanced logistic model. It uses a 600-point anchor at Bad-to-Good odds of 1:50 and 20 points to double the odds. These are score-scale conventions, not lending thresholds.

Notebook 09 defines five risk bands from training out-of-fold predictions. Holdout observed default rates rose from 0.50% in the Very Low band to 23.94% in the Very High band. Its approval and rejection cutoffs are illustrative sample calculations, not a recommended lending policy. Expected-loss examples use hypothetical LGD and EAD assumptions because neither is available in the dataset.

## Scope and limitations

The analysis is based on one supplied dataset and one stratified random split. It has no time-based or external validation, observed lending outcomes beyond the target definition, LGD or EAD data, or reject inference. The reported population stability metrics compare training out-of-fold predictions with the random holdout and do not measure production drift. The results are educational and are not sufficient for operational credit decisions.

## License

This project is released under the MIT License.
