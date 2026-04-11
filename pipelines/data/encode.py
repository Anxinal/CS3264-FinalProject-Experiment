from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from project_paths import DRUG_FINGERPRINTS_NPZ, DRUG_SMILES_JSON, ensure_standard_dirs


def smiles_to_maccs(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


def main():
    ensure_standard_dirs()
    with open(DRUG_SMILES_JSON, "r", encoding="utf-8") as handle:
        drug_smiles = json.load(handle)

    drug_fingerprints = {}
    for drug_id, smiles in drug_smiles.items():
        fp = smiles_to_maccs(smiles)
        if fp is not None:
            drug_fingerprints[drug_id] = fp

    np.savez(
        DRUG_FINGERPRINTS_NPZ,
        ids=list(drug_fingerprints.keys()),
        fps=np.array(list(drug_fingerprints.values())),
    )

    example_id = list(drug_fingerprints.keys())[0]
    print(f"Fingerprint shape: {drug_fingerprints[example_id].shape}")
    print(f"Saved fingerprints to: {DRUG_FINGERPRINTS_NPZ}")


if __name__ == "__main__":
    main()
