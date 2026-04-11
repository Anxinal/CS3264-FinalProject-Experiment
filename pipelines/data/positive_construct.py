from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd

from project_paths import (
    DRUG_FINGERPRINTS_NPZ,
    POSITIVE_SAMPLES_NPZ,
    RAW_DDI_TSV,
    ensure_standard_dirs,
)


def main():
    ensure_standard_dirs()
    data = np.load(DRUG_FINGERPRINTS_NPZ, allow_pickle=True)
    drug_fingerprints = dict(zip(data["ids"], data["fps"]))
    df = pd.read_csv(RAW_DDI_TSV, sep="\t", header=None, names=["drug1", "drug2"])

    positive_samples = []
    for _, row in df.iterrows():
        d1, d2 = row["drug1"], row["drug2"]
        if d1 not in drug_fingerprints or d2 not in drug_fingerprints:
            continue
        fp1 = drug_fingerprints[d1]
        fp2 = drug_fingerprints[d2]
        combined = np.concatenate([fp1, fp2])
        positive_samples.append((combined, 1))

    X_pos = np.array([s[0] for s in positive_samples])
    y_pos = np.array([s[1] for s in positive_samples])
    np.savez(POSITIVE_SAMPLES_NPZ, X=X_pos, y=y_pos)
    print(f"Positive sample count: {len(positive_samples)}")
    print(f"Saved to: {POSITIVE_SAMPLES_NPZ}")


if __name__ == "__main__":
    main()
