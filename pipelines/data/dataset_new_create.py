from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import random

import numpy as np
from sklearn.model_selection import train_test_split

from project_paths import (
    NEGATIVE_ADD_SAMPLES_NPZ,
    POSITIVE_ADD_SAMPLES_NPZ,
    X_TEST_NPY,
    X_TRAIN_NPY,
    X_VAL_NPY,
    Y_TEST_NPY,
    Y_TRAIN_NPY,
    Y_VAL_NPY,
    ensure_standard_dirs,
)


def main():
    ensure_standard_dirs()
    pos_data = np.load(POSITIVE_ADD_SAMPLES_NPZ)
    neg_data = np.load(NEGATIVE_ADD_SAMPLES_NPZ)

    positive_samples = list(zip(pos_data["X"], pos_data["y"]))
    negative_samples = list(zip(neg_data["X"], neg_data["y"]))
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)

    X = np.array([s[0] for s in all_samples])
    y = np.array([s[1] for s in all_samples])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"Additive split shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    np.save(X_TRAIN_NPY, X_train)
    np.save(Y_TRAIN_NPY, y_train)
    np.save(X_VAL_NPY, X_val)
    np.save(Y_VAL_NPY, y_val)
    np.save(X_TEST_NPY, X_test)
    np.save(Y_TEST_NPY, y_test)
    print(f"Saved additive dataset splits to: {X_TRAIN_NPY.parent}")


if __name__ == "__main__":
    main()
