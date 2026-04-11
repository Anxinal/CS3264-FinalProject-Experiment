from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np

from project_paths import DRUG_FINGERPRINTS_NPZ


def main():
    data = np.load(DRUG_FINGERPRINTS_NPZ, allow_pickle=True)
    drug_fingerprints = dict(zip(data["ids"], data["fps"]))
    print(f"Loaded fingerprints for {len(drug_fingerprints)} drugs")
    print(f"Fingerprint shape: {list(drug_fingerprints.values())[0].shape}")


if __name__ == "__main__":
    main()
