# Multi-Timescale Machine Learning for Predicting Kidney Damage in Primary Hyperuricemia

This repository provides the reproducible analysis pipeline, synthetic demonstration data, and model configurations for the manuscript:  
**"Development and validation of machine learning-based models for kidney damage in primary hyperuricemia at three prediction horizons"**.

---

## Repository Contents

1. **`hyperuricemia_kidney-damage.py`**: The complete end-to-end Python pipeline (data preprocessing, MICE imputation, Boruta feature selection across multiple seeds, Optuna hyperparameter optimization, model evaluation, and SHAP interpretability).
2. **`Synthetic data.xlsx`**: An anonymized synthetic dataset matching the schema and statistical distributions of the study cohort, enabling immediate pipeline demonstration without privacy concerns.
3. **`Abbreviations of variables.xlsx`**: Data dictionary mapping variable codes and clinical abbreviations to standard feature names.
4. **`Hyperparameter search spaces and optimal hyperparameters.xlsx`**: Detailed documentation of Optuna search spaces and final optimal hyperparameter configurations for all models.
5. **`requirements.txt`**: List of required Python packages and dependencies.

---

## Installation

Install all required dependencies via pip:
```bash
pip install -r requirements.txt
```
(Developed and tested under Python >= 3.9)

## Execution Guide
1.Open hyperuricemia_kidney-damage.py and set the desktop path variable to your local directory containing the project files.

2.Configure the pipeline switch USE_EXISTING_IMPUTED:

USE_EXISTING_IMPUTED = False: Executes the full raw pipeline from scratch (feature quality filtering, 7:3 stratified train-test splitting, iterative Random Forest MICE imputation, and collinearity screening).

USE_EXISTING_IMPUTED = True: Resumes execution directly using the pre-generated imputed dataset (2.train_dataset_imputed.xlsx), proceeding straight to Boruta feature selection, Optuna tuning, model evaluation, and SHAP attribution.

3.Run the script:
```bash
python hyperuricemia_kidney-damage.py
```
Note: All stochastic operations (splitting, cross-validation, Boruta, and model training) are anchored to a fixed random seed (RANDOM_STATE = 42) to guarantee exact computational reproducibility.

## Data Availability and Ethical Statement
Due to patient privacy protection and institutional ethical regulations (approved by the Medical Ethics Committee of Sichuan Academy of Medical Sciences & Sichuan Provincial People's Hospital), the raw clinical records cannot be publicly deposited.

1.Pipeline Verification: A fully de-identified Synthetic data.xlsx is provided in this repository to facilitate code verification and reproduce the analytical workflow.

2.Access to Real-World Data: De-identified individual participant data can be made available to qualified researchers upon reasonable academic request to the corresponding author, subject to institutional ethics approval.

## License
This project is licensed under the MIT License.
