# Guided TRDP Implementation Plan

## Goal

This round aims to:

- turn the **two-stage RF (SHAP-guided feature sampling)** idea into a reproducible experiment,
- test whether it improves TRDP support over baseline RF (especially **TRDP–SHAP overlap/Jaccard** and readability),
- preserve core Random Forest stochasticity (not hard feature selection).

---

## Phase Overview

## Phase 0 - Lock protocol first

- Choose one fixed data pipeline (either add-167 or concat-324), do not mix in one experiment.
- Keep train/validation/test split fixed.
- Use fixed seeds (at least 3 seeds, averaged).
- Define success criteria:
  - no significant predictive degradation (small AUC/F1 fluctuation is acceptable),
  - TRDP–SHAP Jaccard improves,
  - TRDP path length/readability improves.

**Deliverable**

- `experiment_protocol.md` (one-page spec)

---

## Phase 1 - Baseline RF + SHAP (stage-1 forest)

- Train baseline RF with uniform feature sampling.
- Run SHAP on the trained baseline model (prefer train-fold internal data or validation subset).
- Export global feature importance and normalize it into a sampling prior.

**Leakage rule**

- SHAP used for stage-2 guidance must be computed inside training folds only.
- Validation/test should be used strictly for evaluation.

**Deliverables**

- `rf_stage1_model.pkl`
- `stage1_shap_importance.json`
- baseline metrics table (AUC/F1/Precision/Recall + confusion matrix/accuracy)

---

## Phase 2 - SHAP-guided RF (stage-2 forest)

- Build weighted sampling prior from stage-1 SHAP:

  - `p_i = (importance_i + epsilon)^alpha / sum(...)`
  - `epsilon > 0` keeps non-zero probability for low-importance features
  - `alpha` controls guidance strength (suggested: 0.3 / 0.7 / 1.0; `alpha=0` equals uniform)

- At each split, still sample `m=max_features` candidates, but sample by `p_i` instead of uniform.

**Implementation note**

- sklearn RF does not natively support weighted candidate-feature sampling per split.
- You need either:
  - a custom split-candidate sampling implementation, or
  - another framework that supports feature weights (with a clear explanation of trade-offs).

**Deliverables**

- `rf_stage2_guided_model.pkl`
- `guided_sampling_config.json` (alpha, epsilon, m, seed, pipeline id)

---

## Phase 3 - Comparative evaluation (core)

Evaluate baseline vs guided on the same split/sample set:

### Predictive performance

- AUC, F1, Precision, Recall, Accuracy, Confusion Matrix

### Interpretability consistency

- TRDP top-K features vs SHAP top-N features
- Jaccard, overlap count, optional rank correlation

### TRDP readability

- average path length
- case-level readability checks (does top-1 path include chemically meaningful key features)

**Deliverables**

- `comparison_metrics.csv`
- `interpretability_comparison.md`
- 3-5 case explanation outputs suitable for report inclusion

---

## Minimal ablation matrix

- `alpha`: `[0, 0.3, 0.7, 1.0]` (`0` = baseline control)
- `top_k` (TRDP): `[1, 3]`
- `top_n` (SHAP): `[10, 20]`
- `seed`: `[42, 52, 62]`

This is enough to answer:

- whether Jaccard increases monotonically with guidance strength,
- whether there is an interpretability-performance trade-off turning point.

---

## Risks and mitigations

### Risk 1 - Implementation complexity

- Weighted candidate-feature sampling is not one-line in sklearn.
- Mitigation: deliver a minimal working version first (single alpha, fixed seed), then expand.

### Risk 2 - Better Jaccard but worse predictive performance

- Mitigation: keep baseline as default predictor, guided model as explanation-enhanced mode.

### Risk 3 - Report/code mismatch

- Mitigation: auto-log all configs and outputs to JSON/CSV; avoid manual metric copying.

---

## Report integration guidance

Add a concise subsection in Method/Interpretability:

- "Stage-1 RF (uniform) -> SHAP importance"
- "Stage-2 RF (weighted feature sampling with non-zero floor)"
- explicitly state: "stochasticity preserved; not deterministic feature selection"

In Results, add one compact table:

- `Predictive metrics + Jaccard + Avg path length`
