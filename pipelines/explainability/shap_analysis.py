from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pickle

import matplotlib.pyplot as plt
import numpy as np
import shap

from project_paths import (
    RF_MODEL_PKL,
    SHAP_BAR_PNG,
    SHAP_SUMMARY_PNG,
    SHAP_WATERFALL_PNG,
    X_TEST_NPY,
    Y_TEST_NPY,
    ensure_standard_dirs,
)

ensure_standard_dirs()
X_test = np.load(X_TEST_NPY)
y_test = np.load(Y_TEST_NPY)

with open(RF_MODEL_PKL, "rb") as f:
    rf = pickle.load(f)

explainer = shap.TreeExplainer(rf)
X_sample = X_test[:50]
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_pos = shap_values[1]
    expected_val = explainer.expected_value[1]
else:
    shap_pos = shap_values[:, :, 1]
    expected_val = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value

fp_dim = X_test.shape[1] // 2
maccs_labels = [f"MACCS_{i}" for i in range(fp_dim)]
feature_names_a = [f"DrugA_{label}" for label in maccs_labels]
feature_names_b = [f"DrugB_{label}" for label in maccs_labels]
feature_names = feature_names_a + feature_names_b

plt.figure()
shap.summary_plot(shap_pos, X_sample, feature_names=feature_names, max_display=20, show=False)
plt.tight_layout()
plt.savefig(SHAP_SUMMARY_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {SHAP_SUMMARY_PNG}")

plt.figure()
shap.summary_plot(
    shap_pos,
    X_sample,
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False,
)
plt.tight_layout()
plt.savefig(SHAP_BAR_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {SHAP_BAR_PNG}")

n_sample = X_sample.shape[0]
correct_positive_idx = np.where(
    (y_test[:n_sample] == 1) & ((shap_pos.sum(axis=1) + expected_val) > 0.5)
)[0][0]

shap.waterfall_plot(
    shap.Explanation(
        values=shap_pos[correct_positive_idx],
        base_values=expected_val,
        data=X_sample[correct_positive_idx],
        feature_names=feature_names,
    ),
    max_display=15,
    show=False,
)
plt.tight_layout()
plt.savefig(SHAP_WATERFALL_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {SHAP_WATERFALL_PNG}")

mean_abs_shap = np.abs(shap_pos).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[::-1][:10]
print("\nTop 10 features by mean absolute SHAP:")
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")
