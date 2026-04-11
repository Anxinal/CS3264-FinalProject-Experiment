from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import random

import numpy as np
import pandas as pd

from project_paths import (
    DRUG_FINGERPRINTS_NPZ,
    NEGATIVE_SAMPLES_NPZ,
    POSITIVE_SAMPLES_NPZ,
    RAW_DDI_TSV,
    ensure_standard_dirs,
)


def main():
    ensure_standard_dirs()
    data = np.load(DRUG_FINGERPRINTS_NPZ, allow_pickle=True)
    drug_fingerprints = dict(zip(data["ids"], data["fps"]))
    pos_data = np.load(POSITIVE_SAMPLES_NPZ)
    positive_samples = list(zip(pos_data["X"], pos_data["y"]))
    df = pd.read_csv(RAW_DDI_TSV, sep="\t", header=None, names=["drug1", "drug2"])

    drug_ids = list(drug_fingerprints.keys())
    positive_pairs = set()
    for _, row in df.iterrows():
        positive_pairs.add((row["drug1"], row["drug2"]))
        positive_pairs.add((row["drug2"], row["drug1"]))

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
        combined = np.concatenate([fp1, fp2])
        negative_samples.append((combined, 0))

    X_neg = np.array([s[0] for s in negative_samples])
    y_neg = np.array([s[1] for s in negative_samples])
    np.savez(NEGATIVE_SAMPLES_NPZ, X=X_neg, y=y_neg)
    print(f"Negative sample count: {len(negative_samples)}")
    print(f"Saved to: {NEGATIVE_SAMPLES_NPZ}")


if __name__ == "__main__":
    main()
