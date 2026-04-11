from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import json
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from project_paths import (
    LR_MODEL_PKL,
    LR_RESULTS_JSON,
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

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_pred, y_prob, "Logistic Regression")

    results = {
        "Logistic Regression": {
            "AUC-ROC": roc_auc_score(y_test, y_prob),
            "F1": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
        }
    }
    with open(LR_RESULTS_JSON, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with open(LR_MODEL_PKL, "wb") as handle:
        pickle.dump(lr, handle)
    print(f"Saved metrics to: {LR_RESULTS_JSON}")
    print(f"Saved model to: {LR_MODEL_PKL}")


if __name__ == "__main__":
    main()
