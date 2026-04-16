"""
Qualitative interpretability comparison for CS3264 Final Report.

For 3 true-positive validation samples, produces side-by-side explanations from:
  (A) SHAP     — per-feature attribution scores (baseline RF)
  (B) TRDP     — top-1 decision path from baseline RF
  (C) TRDP-G   — top-1 decision path from SHAP-guided RF (alpha=1.0, T=200, d=10)

Output:
  artifacts/metrics/interpretability_comparison.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

GUIDED_DIR = Path(__file__).resolve().parent / "guided_trdp"
if str(GUIDED_DIR) not in sys.path:
    sys.path.append(str(GUIDED_DIR))

import numpy as np
import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from project_paths import (
    MACCS_REFERENCE_XLSX,
    METRICS_DIR,
    X_TRAIN_NPY,
    X_VAL_NPY,
    Y_TRAIN_NPY,
    Y_VAL_NPY,
    ensure_standard_dirs,
)
from guided_forest import GuidedRandomForest
from trdp.trdp_analysis import build_feature_names, extract_path_conditions

# ── config ─────────────────────────────────────────────────────────────────
N_ESTIMATORS = 200
MAX_DEPTH    = 20
ALPHA        = 1.0
SEED         = 42
SHAP_SAMPLES = 300
TOP_K_SHAP   = 5   # top SHAP features to show per sample
N_SAMPLES    = 3   # number of val samples to explain

EPSILON  = 1e-8

# ── data ───────────────────────────────────────────────────────────────────
ensure_standard_dirs()
X_train = np.load(X_TRAIN_NPY)
y_train = np.load(Y_TRAIN_NPY)
X_val   = np.load(X_VAL_NPY)
y_val   = np.load(Y_VAL_NPY)

N_FEAT   = X_train.shape[1]
feat_names = build_feature_names(N_FEAT)   # returns Feature_0 … Feature_166

print(f"Data: train={X_train.shape}, val={X_val.shape}")


# ── MACCS reference ────────────────────────────────────────────────────────
def load_maccs_ref(xlsx_path: Path) -> dict[int, dict]:
    """Return {bit_position: {short_label, description}} from the xlsx."""
    if not xlsx_path.exists():
        return {}
    df = pd.read_excel(xlsx_path)
    ref: dict[int, dict] = {}
    for _, row in df.iterrows():
        try:
            bit = int(row["Bit Position"])
        except Exception:
            continue
        ref[bit] = {
            "short_label": str(row.get("Short Label", "")).strip(),
            "description": str(row.get("Human-Readable Description", "")).strip(),
        }
    return ref

maccs_ref = load_maccs_ref(MACCS_REFERENCE_XLSX)

def feat_description(feature_idx: int) -> str:
    """Feature index (0-based) → human-readable MACCS description."""
    # RDKit MACCS: index 0 is unused; indices 1-166 correspond to MACCS keys 1-166
    bit = feature_idx  # direct correspondence for additive 167-dim data
    entry = maccs_ref.get(bit)
    if entry:
        return f"Key {bit}: {entry['description']}"
    return f"Key {bit}: (no description)"


# ── Stage 1: baseline RF ───────────────────────────────────────────────────
print(f"\n[1] Training baseline RF (T={N_ESTIMATORS}, max_depth={MAX_DEPTH})...")
baseline = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=SEED, n_jobs=-1
)
baseline.fit(X_train, y_train)
val_proba_base = baseline.predict_proba(X_val)[:, 1]
print(f"    Val AUC (baseline): {roc_auc_score(y_val, val_proba_base):.4f}")

# ── SHAP importances ───────────────────────────────────────────────────────
print(f"[2] Computing SHAP importances on {SHAP_SAMPLES} val samples...")
shap_sample = X_val[:SHAP_SAMPLES]
explainer   = shap.TreeExplainer(baseline)
shap_vals   = explainer.shap_values(shap_sample)
if isinstance(shap_vals, list):
    shap_pos = shap_vals[1]
else:
    shap_pos = shap_vals[:, :, 1]
importances = np.abs(shap_pos).mean(axis=0)  # (n_features,)

# ── Stage 2: guided RF ─────────────────────────────────────────────────────
print(f"[3] Training guided RF (T={N_ESTIMATORS}, max_depth={MAX_DEPTH}, α={ALPHA})...")
pool_size = int(np.sqrt(N_FEAT) * 3)
weights   = np.power(importances + EPSILON, ALPHA)
weights  /= weights.sum()
guided = GuidedRandomForest(
    n_estimators=N_ESTIMATORS,
    tree_feature_pool_size=pool_size,
    max_depth=MAX_DEPTH,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=SEED,
)
guided.fit(X_train, y_train, feature_probabilities=weights)
val_proba_guided = guided.predict_proba(X_val)[:, 1]
print(f"    Val AUC (guided):   {roc_auc_score(y_val, val_proba_guided):.4f}")

# ── Select val samples ─────────────────────────────────────────────────────
# True positives predicted positive by BOTH models for a fair comparison
tp_mask = (
    (y_val == 1) &
    (val_proba_base >= 0.5) &
    (val_proba_guided >= 0.5)
)
candidate_idx = np.where(tp_mask)[0]
# Pick 3 samples with high confidence in baseline for clear explanations
top_conf = candidate_idx[np.argsort(val_proba_base[candidate_idx])[::-1]][:N_SAMPLES]
print(f"[4] Selected val indices: {top_conf.tolist()}")


# ── SHAP per-sample explanation ────────────────────────────────────────────
def shap_explain(sample_idx: int, x: np.ndarray) -> list[dict]:
    """Return top-K SHAP features for a single sample."""
    sv = explainer.shap_values(x.reshape(1, -1))
    if isinstance(sv, list):
        sv_pos = sv[1][0]
    else:
        sv_pos = sv[0, :, 1]
    order = np.argsort(np.abs(sv_pos))[::-1][:TOP_K_SHAP]
    return [
        {
            "rank": int(r + 1),
            "feature_index": int(order[r]),
            "feature_description": feat_description(order[r]),
            "shap_value": float(sv_pos[order[r]]),
            "sample_value": int(x[order[r]]),
        }
        for r in range(len(order))
    ]


# ── TRDP baseline explanation ──────────────────────────────────────────────
def trdp_explain_baseline(x: np.ndarray) -> dict:
    """Top-1 positive-voting tree path from baseline RF."""
    best = None
    for est in baseline.estimators_:
        conds, leaf_id, counts = extract_path_conditions(est, x, feat_names)
        total = float(np.sum(counts))
        pos_p = float(counts[1] / total) if total > 0 else 0.0
        if pos_p > 0.5 and (best is None or pos_p > best["leaf_confidence"]):
            best = {"leaf_confidence": pos_p, "conditions": conds}
    if best is None:
        return {"leaf_confidence": None, "conditions": []}
    readable = []
    for c in best["conditions"]:
        readable.append({
            "feature_index": c["feature_index"],
            "feature_description": feat_description(c["feature_index"]),
            "operator": c["operator"],
            "threshold": c["threshold"],
            "sample_value": c["sample_value"],
        })
    return {"leaf_confidence": best["leaf_confidence"], "conditions": readable}


# ── TRDP guided explanation ────────────────────────────────────────────────
def trdp_explain_guided(x: np.ndarray) -> dict:
    """Top-1 positive-voting tree path from guided RF."""
    best = None
    for tree_obj in guided.trees_:
        local = x[tree_obj.feature_indices].reshape(1, -1)
        leaf_id = int(tree_obj.estimator.apply(local)[0])
        counts = tree_obj.estimator.tree_.value[leaf_id][0]
        total  = float(np.sum(counts))
        pos_p  = float(counts[1] / total) if (total > 0 and len(counts) > 1) else 0.0
        if pos_p > 0.5 and (best is None or pos_p > best["leaf_confidence"]):
            best = {"leaf_confidence": pos_p, "tree_obj": tree_obj}
    if best is None:
        return {"leaf_confidence": None, "conditions": []}

    tree_obj = best["tree_obj"]
    local    = x[tree_obj.feature_indices].reshape(1, -1)
    node_ind = tree_obj.estimator.decision_path(local)
    node_ids = node_ind.indices[node_ind.indptr[0]:node_ind.indptr[1]]
    t = tree_obj.estimator.tree_

    conditions = []
    for nid in node_ids:
        lc, rc = t.children_left[nid], t.children_right[nid]
        if lc == rc:  # leaf
            continue
        local_fi  = int(t.feature[nid])
        global_fi = int(tree_obj.feature_indices[local_fi])
        val       = float(x[global_fi])
        thr       = float(t.threshold[nid])
        op        = "<=" if val <= thr else ">"
        conditions.append({
            "feature_index": global_fi,
            "feature_description": feat_description(global_fi),
            "operator": op,
            "threshold": thr,
            "sample_value": val,
        })

    return {"leaf_confidence": best["leaf_confidence"], "conditions": conditions}


# ── Run for selected samples ───────────────────────────────────────────────
results = []
for idx in top_conf:
    x    = X_val[idx]
    true = int(y_val[idx])
    print(f"\n  Sample val[{idx}]  y_true={true}  "
          f"p_base={val_proba_base[idx]:.3f}  p_guided={val_proba_guided[idx]:.3f}")

    shap_out   = shap_explain(idx, x)
    trdp_base  = trdp_explain_baseline(x)
    trdp_guide = trdp_explain_guided(x)

    results.append({
        "val_index": int(idx),
        "y_true": true,
        "p_baseline": float(val_proba_base[idx]),
        "p_guided":   float(val_proba_guided[idx]),
        "shap": shap_out,
        "trdp_baseline": trdp_base,
        "trdp_guided":   trdp_guide,
    })

    def safe(s: str) -> str:
        return s.encode("ascii", "replace").decode("ascii")

    print(f"  SHAP top-{TOP_K_SHAP}:")
    for s in shap_out:
        print(f"    [{s['rank']}] {safe(s['feature_description'])}  "
              f"phi={s['shap_value']:+.4f}  value={s['sample_value']}")
    print(f"  TRDP baseline  (conf={trdp_base['leaf_confidence']:.3f}, "
          f"depth={len(trdp_base['conditions'])}):")
    for i, c in enumerate(trdp_base["conditions"], 1):
        print(f"    [{i}] {safe(c['feature_description'])}  {c['operator']} {c['threshold']:.1f}  (val={c['sample_value']})")
    print(f"  TRDP guided    (conf={trdp_guide['leaf_confidence']:.3f}, "
          f"depth={len(trdp_guide['conditions'])}):")
    for i, c in enumerate(trdp_guide["conditions"], 1):
        print(f"    [{i}] {safe(c['feature_description'])}  {c['operator']} {c['threshold']:.1f}  (val={c['sample_value']})")


# ── save ───────────────────────────────────────────────────────────────────
out_path = METRICS_DIR / "interpretability_comparison.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {out_path}")
