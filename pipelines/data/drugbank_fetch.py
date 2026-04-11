from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import json
import time

import pandas as pd
import requests

from project_paths import DRUG_SMILES_JSON, RAW_DDI_TSV


def get_smiles_from_pubchem(drugbank_id: str):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/"
        f"{drugbank_id}/property/IsomericSMILES,CanonicalSMILES,SMILES/JSON"
    )
    response = requests.get(url, timeout=30)
    time.sleep(0.2)
    if response.status_code == 200:
        data = response.json()
        props = data["PropertyTable"]["Properties"][0]
        return props.get("IsomericSMILES") or props.get("CanonicalSMILES") or props.get("SMILES")
    return None


def main():
    df = pd.read_csv(RAW_DDI_TSV, sep="\t", header=None, names=["drug1", "drug2"])
    all_drug_ids = set(df["drug1"].tolist() + df["drug2"].tolist())

    drug_smiles = {}
    for drug_id in all_drug_ids:
        smiles = get_smiles_from_pubchem(drug_id)
        if smiles:
            drug_smiles[drug_id] = smiles
        print(f"{drug_id}: {smiles}")

    with open(DRUG_SMILES_JSON, "w", encoding="utf-8") as handle:
        json.dump(drug_smiles, handle)
    print(f"Saved SMILES map to: {DRUG_SMILES_JSON}")


if __name__ == "__main__":
    main()
