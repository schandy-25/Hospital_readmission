# Hospital Readmission Risk Model (Cost-Aware)

This project implements a 30-day hospital readmission risk model on a diabetes hospitalization dataset.

Key features:
- End-to-end ML pipeline (preprocessing + gradient boosting classifier)
- **Cost-aware threshold optimization** for asymmetric clinical costs
- **Interpretability with SHAP** (global feature importance)

## Dataset

- ~100k encounters
- Target: `readmitted` column
  - Positive class: `<30` (readmitted within 30 days)
  - Negative class: `>30` or `NO`
- Positive rate in this setup: ~11%

## Model & Results

Model: `HistGradientBoostingClassifier` with numeric scaling + one-hot encoded categorical features.

On the held-out test set:

- **Validation ROC AUC:** ~0.686  
- **Test ROC AUC:** ~0.673  

Cost model (edit if you change it in code):

- `COST_FN = 10.0` (missing a high-risk patient)
- `COST_FP = 1.0` (flagging a low-risk patient)

Using this cost model, we search thresholds on the validation set and pick the one that minimizes average cost per patient. At the best threshold, we report:

- Threshold: *(fill from terminal, e.g. 0.34)*  
- Avg cost per patient (validation): *(value)*  
- Avg cost per patient (test): *(value)*  
- Sensitivity (recall of positives): *(value)*  
- Specificity (recall of negatives): *(value)*  
- Confusion matrix (test): TP, FP, FN, TN

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/train_cost_aware_model.py --data_path data/diabetes_readmission.csv

