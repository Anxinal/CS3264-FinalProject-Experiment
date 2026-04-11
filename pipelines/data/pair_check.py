from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import pandas as pd

from project_paths import RAW_DDI_TSV


def main():
    df = pd.read_csv(RAW_DDI_TSV, sep="\t", header=None, names=["drug1", "drug2"])
    df_pairs = set(zip(df["drug1"], df["drug2"]))
    symmetric_count = 0
    for d1, d2 in df_pairs:
        if (d2, d1) in df_pairs:
            symmetric_count += 1
    print(f"Symmetric pair count: {symmetric_count}")
    print(f"Total positive pair count: {len(df_pairs)}")


if __name__ == "__main__":
    main()
