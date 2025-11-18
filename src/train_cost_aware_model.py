"""
Hospital Readmission Risk Model (Cost-Aware)
-------------------------------------------
- Train a 30-day readmission classifier
- Optimize decision threshold using a simple cost model
- Use SHAP for global interpretability

Usage:
    python src/train_cost_aware_model.py --data_path data/diabetes_readmission.csv
"""

import argparse
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# --------------------------
# Config: cost model
# --------------------------
# You can tune these based on clinical assumptions
COST_FN = 10.0  # Cost of missing a high-risk patient (false negative)
COST_FP = 1.0   # Cost of flagging a low-risk patient (false positive)


def load_and_preprocess(data_path: str):
    """
    Load the dataset and split into X, y.
    Assumes:
      - 'readmitted' column with values like '<30', '>30', 'NO'
      - We'll define y = 1 if '<30', else 0
    """
    df = pd.read_csv(data_path)

    # Drop obvious ID columns if present
    id_cols = [c for c in ["encounter_id", "patient_nbr"] if c in df.columns]
    df = df.drop(columns=id_cols, errors="ignore")

    if "readmitted" not in df.columns:
        raise ValueError("Expected a 'readmitted' column in the dataset.")

    # Binary target: readmitted within 30 days
    y = (df["readmitted"] == "<30").astype(int)

    X = df.drop(columns=["readmitted"])

    # Split numerical vs categorical by dtype
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    return X, y, numeric_cols, categorical_cols


def build_pipeline(numeric_cols, categorical_cols):
    """
    Build preprocessing + model pipeline.
    - Numeric: StandardScaler
    - Categorical: OneHotEncoder (dense)
    - Model: HistGradientBoostingClassifier (requires dense input)
    """
    numeric_transformer = StandardScaler()
    # Make encoder output dense instead of sparse
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False  # IMPORTANT: dense output for HistGradientBoosting
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.05,
        max_iter=200,
        random_state=42
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return clf



def expected_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    """
    Compute expected misclassification cost given binary predictions.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)

    cost = cost_fn * fn + cost_fp * fp
    return cost / total  # average cost per patient


def find_best_threshold(y_true, y_proba, cost_fn=COST_FN, cost_fp=COST_FP, num_thresh=200):
    """
    Search over thresholds to minimize expected cost.
    Returns:
      best_threshold, best_cost, metrics_at_best (dict)
    """
    thresholds = np.linspace(0.0, 1.0, num_thresh)
    best_t = 0.5
    best_cost = float("inf")
    best_metrics = {}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cost = expected_cost(y_true, y_pred, cost_fn, cost_fp)

        if cost < best_cost:
            best_cost = cost
            best_t = t

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall for positive class
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall for negative class

            best_metrics = {
                "threshold": best_t,
                "cost": best_cost,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }

    return best_t, best_cost, best_metrics


def compute_shap_values(clf: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, max_samples=1000):
    """
    Compute SHAP values for the tree-based model inside the pipeline.
    We'll:
      - Grab the underlying HistGradientBoostingClassifier
      - Transform X through the preprocessor
      - Use TreeExplainer
    """
    # Extract fitted parts
    preprocessor = clf.named_steps["preprocess"]
    model = clf.named_steps["model"]

    # Transform data
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Optionally subsample for SHAP to keep it fast
    if X_test_trans.shape[0] > max_samples:
        idx = np.random.RandomState(42).choice(X_test_trans.shape[0], size=max_samples, replace=False)
        X_shap = X_test_trans[idx]
    else:
        X_shap = X_test_trans

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    return explainer, shap_values, X_shap


def main(args):
    # 1. Load data
    print("Loading data...")
    X, y, numeric_cols, categorical_cols = load_and_preprocess(args.data_path)
    print(f"Dataset shape: X={X.shape}, y={y.shape}, positive rate={y.mean():.3f}")

    # 2. Train/valid/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 3. Build pipeline & train
    print("Building and training model...")
    clf = build_pipeline(numeric_cols, categorical_cols)
    clf.fit(X_train, y_train)

    # 4. Evaluate ROC AUC
    y_valid_proba = clf.predict_proba(X_valid)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    auc_valid = roc_auc_score(y_valid, y_valid_proba)
    auc_test = roc_auc_score(y_test, y_test_proba)
    print(f"Validation ROC AUC: {auc_valid:.3f}")
    print(f"Test ROC AUC:       {auc_test:.3f}")

    # 5. Cost-aware threshold optimization on validation set
    print("\nSearching for cost-optimal threshold (on validation set)...")
    best_t, best_cost, best_metrics = find_best_threshold(
        y_valid, y_valid_proba, COST_FN, COST_FP
    )
    print("Best threshold on validation:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v}")

    # 6. Evaluate on test set using that threshold
    y_test_pred = (y_test_proba >= best_t).astype(int)
    test_cost = expected_cost(y_test, y_test_pred, COST_FN, COST_FP)
    print(f"\nTest avg cost (using threshold={best_t:.3f}): {test_cost:.4f}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_test_pred, digits=3))

    # 7. SHAP interpretability
    print("\nComputing SHAP values for interpretability...")
    explainer, shap_values, X_shap = compute_shap_values(clf, X_train, X_test)

    # SHAP summary plot
    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title("SHAP Summary Plot - Hospital Readmission Model")
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    print("Saved SHAP summary plot to shap_summary.png")

    # Optional: save model with joblib
    if args.save_model_path:
        import joblib
        joblib.dump(clf, args.save_model_path)
        print(f"Saved trained model to {args.save_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/diabetes_readmission.csv",
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="hospital_readmission_model.joblib",
        help="Where to save the trained model (joblib file).",
    )
    args = parser.parse_args()
    main(args)

