from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import random

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from project_paths import (
    DRUG_FINGERPRINTS_NPZ,
    RAW_DDI_TSV,
    X_TEST_NPY,
    X_TRAIN_NPY,
    X_VAL_NPY,
    Y_TEST_NPY,
    Y_TRAIN_NPY,
    Y_VAL_NPY,
)


def main():
    positive_samples = []
    positive_pairs = set()
    df = pd.read_csv(RAW_DDI_TSV, sep="\t", header=None, names=["drug1", "drug2"])
    data = np.load(DRUG_FINGERPRINTS_NPZ, allow_pickle=True)
    drug_fingerprints = dict(zip(data["ids"], data["fps"]))

    for _, row in df.iterrows():
        d1, d2 = row["drug1"], row["drug2"]
        if d1 not in drug_fingerprints or d2 not in drug_fingerprints:
            continue
        fp1 = drug_fingerprints[d1]
        fp2 = drug_fingerprints[d2]
        positive_samples.append((np.concatenate([fp1, fp2]), 1))
        positive_samples.append((np.concatenate([fp2, fp1]), 1))
        positive_pairs.add((d1, d2))
        positive_pairs.add((d2, d1))

    print(f"Reconstructed positive count: {len(positive_samples)}")

    drug_ids = list(drug_fingerprints.keys())
    negative_samples = []
    target_count = len(positive_samples)
    while len(negative_samples) < target_count:
        d1 = random.choice(drug_ids)
        d2 = random.choice(drug_ids)
        if d1 == d2:
            continue
        if (d1, d2) in positive_pairs:
            continue
        fp1 = drug_fingerprints[d1]
        fp2 = drug_fingerprints[d2]
        negative_samples.append((np.concatenate([fp1, fp2]), 0))
        negative_samples.append((np.concatenate([fp2, fp1]), 0))
        positive_pairs.add((d1, d2))
        positive_pairs.add((d2, d1))

    print(f"Reconstructed negative count: {len(negative_samples)}")

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

    selector = VarianceThreshold(threshold=0.0)
    X_train = selector.fit_transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)

    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")

    np.save(X_TRAIN_NPY, X_train)
    np.save(X_VAL_NPY, X_val)
    np.save(X_TEST_NPY, X_test)
    np.save(Y_TRAIN_NPY, y_train)
    np.save(Y_VAL_NPY, y_val)
    np.save(Y_TEST_NPY, y_test)
    print("Reconstructed dataset saved")


if __name__ == "__main__":
    main()
