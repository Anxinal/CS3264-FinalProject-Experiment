"""
Confusion matrix experiment for CS3264 Final Report.

Runs:
  - Logistic Regression (baseline)
  - Random Forest with hyperparameter sweep:
      n_estimators in {50, 100, 200}
      max_depth    in {3, 5, 10, 20}
  - MLP with architecture matching the final report:
      input_dim -> input_dim -> 128 -> 64 -> 1
      (input_dim determined from data; report planned 162, actual 167)

All models are trained on X_train / y_train and evaluated on X_test / y_test.
Results are written to artifacts/metrics/confusion_matrix_results.json.
"""

from pathlib import Path
import sys
import json
import time

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from project_paths import (
    METRICS_DIR,
    X_TRAIN_NPY,
    X_VAL_NPY,
    X_TEST_NPY,
    Y_TRAIN_NPY,
    Y_VAL_NPY,
    Y_TEST_NPY,
    ensure_standard_dirs,
)

# ── data ──────────────────────────────────────────────────────────────────────
ensure_standard_dirs()
X_train = np.load(X_TRAIN_NPY)
y_train = np.load(Y_TRAIN_NPY)
X_val   = np.load(X_VAL_NPY)
y_val   = np.load(Y_VAL_NPY)
X_test  = np.load(X_TEST_NPY)
y_test  = np.load(Y_TEST_NPY)

INPUT_DIM = X_train.shape[1]
print(f"Data loaded — train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")


# ── helpers ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, name: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "model": name,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "auc_roc":   float(roc_auc_score(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall":    float(recall_score(y_true, y_pred)),
    }
    print(
        f"  {name:<45}  "
        f"AUC={metrics['auc_roc']:.4f}  F1={metrics['f1']:.4f}  "
        f"Acc={metrics['accuracy']:.4f}  "
        f"TP={tp} TN={tn} FP={fp} FN={fn}"
    )
    return metrics


# ── Logistic Regression ───────────────────────────────────────────────────────
print("\n[1/3] Logistic Regression")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]
lr_result = compute_metrics(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")


# ── Random Forest hyperparameter sweep ───────────────────────────────────────
print("\n[2/3] Random Forest — hyperparameter sweep")
N_ESTIMATORS_LIST = [50, 100, 200]
MAX_DEPTH_LIST    = [3, 5, 10, 20]

rf_results = []
for n_est in N_ESTIMATORS_LIST:
    for max_d in MAX_DEPTH_LIST:
        name = f"RF n_est={n_est} max_depth={max_d}"
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]
        res = compute_metrics(y_test, y_pred, y_prob, name)
        res["n_estimators"] = n_est
        res["max_depth"]    = max_d
        res["train_time_s"] = round(time.time() - t0, 2)
        rf_results.append(res)


# ── MLP ───────────────────────────────────────────────────────────────────────
print(f"\n[3/3] MLP  (architecture: {INPUT_DIM}→{INPUT_DIM}→128→64→1)")

class MLP(nn.Module):
    """Architecture matching the final report (input dim adjusted to actual data)."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze()


X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t   = torch.FloatTensor(X_val)
y_val_t   = torch.FloatTensor(y_val)
X_test_t  = torch.FloatTensor(X_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)

mlp = MLP(INPUT_DIM)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
criterion = nn.BCELoss()

best_val_auc   = 0.0
best_state     = None
EPOCHS         = 50

for epoch in range(EPOCHS):
    mlp.train()
    total_loss = 0.0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        loss = criterion(mlp(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    mlp.eval()
    with torch.no_grad():
        val_probs = mlp(X_val_t).numpy()
        val_auc   = roc_auc_score(y_val, val_probs)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state   = {k: v.clone() for k, v in mlp.state_dict().items()}

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:02d}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  val_auc={val_auc:.4f}")

mlp.load_state_dict(best_state)
print(f"  Best val AUC: {best_val_auc:.4f}")

mlp.eval()
with torch.no_grad():
    y_prob_mlp = mlp(X_test_t).numpy()
y_pred_mlp = (y_prob_mlp > 0.5).astype(int)
mlp_result = compute_metrics(y_test, y_pred_mlp, y_prob_mlp, f"MLP ({INPUT_DIM}→{INPUT_DIM}→128→64→1)")
mlp_result["best_val_auc"] = float(best_val_auc)
mlp_result["input_dim"]    = int(INPUT_DIM)


# ── save ──────────────────────────────────────────────────────────────────────
output = {
    "input_dim": int(INPUT_DIM),
    "logistic_regression": lr_result,
    "random_forest_sweep": rf_results,
    "mlp": mlp_result,
}

out_path = METRICS_DIR / "confusion_matrix_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {out_path}")
