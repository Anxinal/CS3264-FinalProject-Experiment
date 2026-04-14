"""
Two-stage SHAP-guided Random Forest experiment.
  Configuration: T=200, max_depth=10
  Stage 1 — train baseline RF, compute SHAP feature importances.
  Stage 2 — train GuidedRandomForest (same T/depth) with SHAP-weighted
             per-tree feature-pool sampling (alpha sweep: 0.3, 0.7, 1.0).

Saves metrics + confusion matrices to:
  artifacts/metrics/rf_d10_shap_guided_results.json
"""

from __future__ import annotations

from pathlib import Path
import sys
import json

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

GUIDED_DIR = Path(__file__).resolve().parent / "guided_trdp"
if str(GUIDED_DIR) not in sys.path:
    sys.path.append(str(GUIDED_DIR))

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from project_paths import (
    METRICS_DIR,
    X_TEST_NPY,
    X_TRAIN_NPY,
    X_VAL_NPY,
    Y_TEST_NPY,
    Y_TRAIN_NPY,
    ensure_standard_dirs,
)
from guided_forest import GuidedRandomForest

# ── config ─────────────────────────────────────────────────────────────────
N_ESTIMATORS  = 200
MAX_DEPTH     = 10
SEED          = 42
ALPHAS        = [0.3, 0.7, 1.0]
EPSILON       = 1e-8
SHAP_SAMPLES  = 300   # val samples for SHAP (capped for speed)

# ── data ───────────────────────────────────────────────────────────────────
ensure_standard_dirs()
X_train = np.load(X_TRAIN_NPY)
y_train = np.load(Y_TRAIN_NPY)
X_val   = np.load(X_VAL_NPY)
X_test  = np.load(X_TEST_NPY)
y_test  = np.load(Y_TEST_NPY)

N_FEATURES          = X_train.shape[1]
POOL_SIZE           = int(np.sqrt(N_FEATURES) * 3)   # matches existing experiment convention

print(f"Data: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
print(f"Config: T={N_ESTIMATORS}, max_depth={MAX_DEPTH}, pool_size={POOL_SIZE}")


# ── helpers ────────────────────────────────────────────────────────────────
def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, label: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    m = {
        "model":     label,
        "n_estimators": N_ESTIMATORS,
        "max_depth":    MAX_DEPTH,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "auc_roc":   float(roc_auc_score(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall":    float(recall_score(y_true, y_pred)),
    }
    print(
        f"  {label:<45}  AUC={m['auc_roc']:.4f}  F1={m['f1']:.4f}  "
        f"Acc={m['accuracy']:.4f}  TP={tp} TN={tn} FP={fp} FN={fn}"
    )
    return m


# ── Stage 1: baseline RF ───────────────────────────────────────────────────
print(f"\n[Stage 1] Baseline RF (T={N_ESTIMATORS}, max_depth={MAX_DEPTH})")
baseline = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=SEED,
    n_jobs=-1,
)
baseline.fit(X_train, y_train)
y_prob_base = baseline.predict_proba(X_test)[:, 1]
y_pred_base = (y_prob_base >= 0.5).astype(int)
base_metrics = get_metrics(y_test, y_pred_base, y_prob_base, f"Baseline RF (T={N_ESTIMATORS}, d={MAX_DEPTH})")

# ── SHAP importances ───────────────────────────────────────────────────────
print(f"\n[Stage 1] Computing SHAP importances on {SHAP_SAMPLES} val samples...")
shap_sample = X_val[:SHAP_SAMPLES]
explainer   = shap.TreeExplainer(baseline)
shap_vals   = explainer.shap_values(shap_sample)

# shap_values may be ndarray (n, d, 2) or list of two (n, d) arrays
if isinstance(shap_vals, list):
    shap_pos = shap_vals[1]          # positive class
else:
    shap_pos = shap_vals[:, :, 1]   # shape (n, d, 2) → slice class-1

importances = np.abs(shap_pos).mean(axis=0)   # shape (d,)
print(f"  SHAP importances computed — top-5 feature indices: {np.argsort(importances)[::-1][:5].tolist()}")


# ── Stage 2: guided RF sweep ───────────────────────────────────────────────
def compute_probs(imp: np.ndarray, alpha: float) -> np.ndarray:
    w = np.power(imp + EPSILON, alpha)
    return w / w.sum()

guided_results = []
for alpha in ALPHAS:
    label = f"Guided RF (T={N_ESTIMATORS}, d={MAX_DEPTH}, α={alpha})"
    print(f"\n[Stage 2] {label}")
    probs = compute_probs(importances, alpha)
    guided = GuidedRandomForest(
        n_estimators=N_ESTIMATORS,
        tree_feature_pool_size=POOL_SIZE,
        max_depth=MAX_DEPTH,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=SEED,
    )
    guided.fit(X_train, y_train, feature_probabilities=probs)
    y_prob_g = guided.predict_proba(X_test)[:, 1]
    y_pred_g = (y_prob_g >= 0.5).astype(int)
    m = get_metrics(y_test, y_pred_g, y_prob_g, label)
    m["alpha"] = alpha
    guided_results.append(m)


# ── save ───────────────────────────────────────────────────────────────────
output = {
    "description": f"Baseline vs SHAP-guided RF: T={N_ESTIMATORS}, max_depth={MAX_DEPTH}",
    "config": {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "seed": SEED,
        "pool_size": POOL_SIZE,
        "shap_samples": SHAP_SAMPLES,
        "epsilon": EPSILON,
    },
    "baseline": base_metrics,
    "guided_sweep": guided_results,
}

out_path = METRICS_DIR / "rf_d10_shap_guided_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to: {out_path}")
