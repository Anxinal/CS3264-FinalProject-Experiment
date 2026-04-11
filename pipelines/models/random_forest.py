from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import json
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from project_paths import (
    RF_MODEL_PKL,
    RF_RESULTS_JSON,
    X_TEST_NPY,
    X_TRAIN_NPY,
    Y_TEST_NPY,
    Y_TRAIN_NPY,
    ensure_standard_dirs,
)


def evaluate(y_true, y_pred, y_prob, model_name: str):
    print(f"\n{'=' * 40}")
    print(f"  {model_name}")
    print(f"{'=' * 40}")
    print(f"AUC-ROC:   {roc_auc_score(y_true, y_prob):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")


def main():
    ensure_standard_dirs()
    X_train = np.load(X_TRAIN_NPY)
    y_train = np.load(Y_TRAIN_NPY)
    X_test = np.load(X_TEST_NPY)
    y_test = np.load(Y_TEST_NPY)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_pred_rf, y_prob_rf, "Random Forest")

    results = {
        "Random Forest": {
            "AUC-ROC": roc_auc_score(y_test, y_prob_rf),
            "F1": f1_score(y_test, y_pred_rf),
            "Precision": precision_score(y_test, y_pred_rf),
            "Recall": recall_score(y_test, y_pred_rf),
        }
    }
    with open(RF_RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with open(RF_MODEL_PKL, "wb") as handle:
        pickle.dump(rf, handle)
    print(f"Saved metrics to: {RF_RESULTS_JSON}")
    print(f"Saved model to: {RF_MODEL_PKL}")


if __name__ == "__main__":
    main()
