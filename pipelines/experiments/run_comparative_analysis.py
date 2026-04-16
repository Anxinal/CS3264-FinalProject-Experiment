"""
Comparative Analysis & TRDP Rule Accuracy Experiment
=====================================================
CS3264 Final Report — Comparative Analysis + Qualitative Case Study

Experiments:
  1. For anchor samples, find val-set neighbours with L1 distance <= k.
     For each neighbour compute the TRDP path from BOTH the baseline RF and
     the SHAP-guided RF, then report per-model Jaccard and prediction
     consistency --- giving an explicit side-by-side comparison.

  2. Apply the TOP-5-CONDITION TRDP rule for val[ANCHOR_IDX] (from both
     models) as standalone rules to the full validation set.

Outputs:
  artifacts/metrics/comparative_analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
GUIDED_DIR = Path(__file__).resolve().parent / "guided_trdp"
for p in (str(ROOT_DIR), str(GUIDED_DIR)):
    if p not in sys.path:
        sys.path.append(p)

import numpy as np
import shap as shap_lib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# ── config ──────────────────────────────────────────────────────────────────
N_ESTIMATORS   = 200
MAX_DEPTH      = 20
ALPHA          = 1.0
SEED           = 42
SHAP_SAMPLES   = 300
EPSILON        = 1e-8

K_VALUES       = [1, 3, 5, 10, 20, 30]
TOP_CONDITIONS = 5            # conditions used for the standalone-rule test

ANCHOR_IDX    = 5271
EXTRA_ANCHORS = [2823, 5640]

# ── data ────────────────────────────────────────────────────────────────────
ensure_standard_dirs()
X_train = np.load(X_TRAIN_NPY)
y_train = np.load(Y_TRAIN_NPY)
X_val   = np.load(X_VAL_NPY)
y_val   = np.load(Y_VAL_NPY)
N_FEAT  = X_train.shape[1]
feat_names = build_feature_names(N_FEAT)
print(f"Data: train={X_train.shape}  val={X_val.shape}")

# ── Stage 1: baseline RF ────────────────────────────────────────────────────
print(f"\n[1] Training baseline RF (T={N_ESTIMATORS}, d={MAX_DEPTH}, seed={SEED})...")
baseline = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
    random_state=SEED, n_jobs=-1,
)
baseline.fit(X_train, y_train)
val_proba_base = baseline.predict_proba(X_val)[:, 1]
val_pred_base  = (val_proba_base >= 0.5).astype(int)

# ── SHAP importances ────────────────────────────────────────────────────────
print(f"[2] Computing SHAP importances on {SHAP_SAMPLES} val samples...")
explainer   = shap_lib.TreeExplainer(baseline)
shap_vals   = explainer.shap_values(X_val[:SHAP_SAMPLES])
if isinstance(shap_vals, list):
    shap_pos = shap_vals[1]
else:
    shap_pos = shap_vals[:, :, 1]
importances = np.abs(shap_pos).mean(axis=0)

# ── Stage 2: guided RF ──────────────────────────────────────────────────────
print(f"[3] Training guided RF (T={N_ESTIMATORS}, d={MAX_DEPTH}, α={ALPHA})...")
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
val_pred_guided  = (val_proba_guided >= 0.5).astype(int)


# ── TRDP helpers ────────────────────────────────────────────────────────────
def get_top1_baseline(x: np.ndarray) -> tuple[set[int], list[dict], float]:
    best_conf, best_conds = -1.0, []
    for est in baseline.estimators_:
        conds, _, counts = extract_path_conditions(est, x, feat_names)
        total = float(np.sum(counts))
        pos_p = float(counts[1] / total) if total > 0 else 0.0
        if pos_p > 0.5 and pos_p > best_conf:
            best_conf, best_conds = pos_p, conds
    return {c["feature_index"] for c in best_conds}, best_conds, best_conf


def get_top1_guided(x: np.ndarray) -> tuple[set[int], list[dict], float]:
    best_conf, best_conds = -1.0, []
    for tree_obj in guided.trees_:
        local  = x[tree_obj.feature_indices].reshape(1, -1)
        leaf_id = int(tree_obj.estimator.apply(local)[0])
        counts  = tree_obj.estimator.tree_.value[leaf_id][0]
        total   = float(np.sum(counts))
        pos_p   = float(counts[1] / total) if (total > 0 and len(counts) > 1) else 0.0
        if pos_p > 0.5 and pos_p > best_conf:
            best_conf = pos_p
            # reconstruct conditions in global feature indices
            node_ind = tree_obj.estimator.decision_path(local)
            node_ids = node_ind.indices[node_ind.indptr[0]:node_ind.indptr[1]]
            t = tree_obj.estimator.tree_
            conds = []
            for nid in node_ids:
                if t.children_left[nid] == t.children_right[nid]:
                    continue
                local_fi  = int(t.feature[nid])
                global_fi = int(tree_obj.feature_indices[local_fi])
                val_      = float(x[global_fi])
                thr       = float(t.threshold[nid])
                conds.append({
                    "feature_index": global_fi,
                    "operator":      "<=" if val_ <= thr else ">",
                    "threshold":     thr,
                    "sample_value":  val_,
                })
            best_conds = conds
    return {c["feature_index"] for c in best_conds}, best_conds, best_conf


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return float(len(a & b) / u) if u > 0 else 1.0


# ── Experiment 1: comparative analysis ──────────────────────────────────────
print("\n[4] Running comparative analysis...")

def analyse_anchor(anchor_idx: int) -> dict:
    x_a = X_val[anchor_idx]

    # anchor TRDP from both models
    a_feats_base, a_conds_base, a_conf_base   = get_top1_baseline(x_a)
    a_feats_guid, a_conds_guid, a_conf_guided = get_top1_guided(x_a)

    l1_dists = np.sum(np.abs(X_val - x_a), axis=1)

    by_k: dict[int, dict] = {}
    for k in K_VALUES:
        mask  = (l1_dists <= k) & (np.arange(len(X_val)) != anchor_idx)
        idxs  = np.where(mask)[0]
        n     = int(len(idxs))

        if n == 0:
            by_k[k] = {"k": k, "n_neighbours": 0,
                       "baseline": None, "guided": None,
                       "cross_model_anchor": None}
            continue

        j_base_list, j_guid_list, j_cross_list = [], [], []
        c_base_list, c_guid_list = [], []

        neighbour_rows = []
        for nidx in idxs:
            x_n = X_val[nidx]
            n_feats_base, _, _ = get_top1_baseline(x_n)
            n_feats_guid, _, _ = get_top1_guided(x_n)

            jb = jaccard(a_feats_base, n_feats_base)   # baseline vs baseline
            jg = jaccard(a_feats_guid, n_feats_guid)   # guided   vs guided
            # cross: anchor baseline path vs anchor guided path on the same neighbour
            jx = jaccard(n_feats_base, n_feats_guid)

            cb = int(val_pred_base[nidx]   == val_pred_base[anchor_idx])
            cg = int(val_pred_guided[nidx] == val_pred_guided[anchor_idx])

            j_base_list.append(jb); j_guid_list.append(jg); j_cross_list.append(jx)
            c_base_list.append(cb); c_guid_list.append(cg)
            neighbour_rows.append({
                "val_index": int(nidx), "l1": int(l1_dists[nidx]),
                "y_true": int(y_val[nidx]),
                "jaccard_baseline": jb, "jaccard_guided": jg,
                "jaccard_cross_neighbour": jx,
                "pred_same_baseline": cb, "pred_same_guided": cg,
            })

        by_k[k] = {
            "k": k, "n_neighbours": n,
            "baseline": {
                "mean_jaccard": float(np.mean(j_base_list)),
                "std_jaccard":  float(np.std(j_base_list)),
                "pred_consistency": float(np.mean(c_base_list)),
            },
            "guided": {
                "mean_jaccard": float(np.mean(j_guid_list)),
                "std_jaccard":  float(np.std(j_guid_list)),
                "pred_consistency": float(np.mean(c_guid_list)),
            },
            # Jaccard between baseline and guided paths on the SAME neighbour
            "cross_model_neighbour_mean_jaccard": float(np.mean(j_cross_list)),
            "neighbours": neighbour_rows,
        }
        print(f"  anchor={anchor_idx}  k={k:2d}: n={n:3d} | "
              f"Jac_base={np.mean(j_base_list):.3f}  "
              f"Jac_guided={np.mean(j_guid_list):.3f}  "
              f"Jac_cross={np.mean(j_cross_list):.3f}  "
              f"C_base={np.mean(c_base_list):.3f}  C_guided={np.mean(c_guid_list):.3f}")

    return {
        "anchor_idx":     anchor_idx,
        "y_true":         int(y_val[anchor_idx]),
        "baseline": {
            "proba":          float(val_proba_base[anchor_idx]),
            "leaf_conf":      float(a_conf_base),
            "trdp_depth":     len(a_conds_base),
            "trdp_features":  sorted(a_feats_base),
        },
        "guided": {
            "proba":          float(val_proba_guided[anchor_idx]),
            "leaf_conf":      float(a_conf_guided),
            "trdp_depth":     len(a_conds_guid),
            "trdp_features":  sorted(a_feats_guid),
        },
        # Jaccard between baseline and guided paths on the ANCHOR itself
        "anchor_cross_model_jaccard": jaccard(a_feats_base, a_feats_guid),
        "by_k": by_k,
    }

ca_results = [analyse_anchor(a) for a in [ANCHOR_IDX] + EXTRA_ANCHORS]


# ── Experiment 2: top-5-condition rule accuracy ──────────────────────────────
print(f"\n[5] Top-{TOP_CONDITIONS}-condition rule accuracy for val[{ANCHOR_IDX}]...")

def apply_rule(x: np.ndarray, conditions: list[dict]) -> int:
    for c in conditions:
        v = float(x[c["feature_index"]])
        if c["operator"] == "<=" and not (v <= c["threshold"]):
            return 0
        if c["operator"] == ">"  and not (v >  c["threshold"]):
            return 0
    return 1

def rule_stats(conds: list[dict], label: str) -> dict:
    top_conds = conds[:TOP_CONDITIONS]
    preds = np.array([apply_rule(X_val[i], top_conds) for i in range(len(X_val))])
    y = y_val.astype(int)
    tp = int(np.sum((preds==1)&(y==1))); fp = int(np.sum((preds==1)&(y==0)))
    tn = int(np.sum((preds==0)&(y==0))); fn = int(np.sum((preds==0)&(y==1)))
    acc  = float(accuracy_score(y, preds))
    prec = float(precision_score(y, preds, zero_division=0))
    rec  = float(recall_score(y, preds, zero_division=0))
    f1   = float(f1_score(y, preds, zero_division=0))
    n_flagged = int(np.sum(preds==1))
    print(f"  [{label}] top-{TOP_CONDITIONS} rule: flagged={n_flagged}, "
          f"TP={tp} FP={fp} TN={tn} FN={fn} | "
          f"Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
    return {
        "model": label,
        "n_conditions_used": TOP_CONDITIONS,
        "n_flagged": n_flagged, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "conditions": [
            {"node": i+1, "feature_index": c["feature_index"],
             "operator": c["operator"], "threshold": c["threshold"]}
            for i, c in enumerate(top_conds)
        ],
    }

_, base_conds_anchor, _ = get_top1_baseline(X_val[ANCHOR_IDX])
_, guid_conds_anchor, _ = get_top1_guided(X_val[ANCHOR_IDX])

rule_results = [
    rule_stats(base_conds_anchor, "baseline"),
    rule_stats(guid_conds_anchor, "guided"),
]

# ── Save ─────────────────────────────────────────────────────────────────────
output = {
    "comparative_analysis": ca_results,
    "trdp_rule_accuracy":   rule_results,
}
out_path = METRICS_DIR / "comparative_analysis.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved → {out_path}")
