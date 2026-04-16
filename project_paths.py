from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
EXPLANATIONS_DIR = ARTIFACTS_DIR / "explanations"

RAW_DDI_TSV = ROOT_DIR / "ChCh-Miner_durgbank-chem-chem.tsv" / "ChCh-Miner_durgbank-chem-chem.tsv"
DRUG_SMILES_JSON = ROOT_DIR / "drug_smiles.json"
DRUG_FINGERPRINTS_NPZ = PROCESSED_DIR / "drug_fingerprints.npz"

POSITIVE_ADD_SAMPLES_NPZ = PROCESSED_DIR / "positive_add_samples.npz"
NEGATIVE_ADD_SAMPLES_NPZ = PROCESSED_DIR / "negative_add_samples.npz"

X_TRAIN_NPY = PROCESSED_DIR / "X_train.npy"
Y_TRAIN_NPY = PROCESSED_DIR / "y_train.npy"
X_VAL_NPY = PROCESSED_DIR / "X_val.npy"
Y_VAL_NPY = PROCESSED_DIR / "y_val.npy"
X_TEST_NPY = PROCESSED_DIR / "X_test.npy"
Y_TEST_NPY = PROCESSED_DIR / "y_test.npy"

RF_MODEL_PKL = MODELS_DIR / "rf_model.pkl"
LR_MODEL_PKL = MODELS_DIR / "lr_model.pkl"
MLP_MODEL_PTH = MODELS_DIR / "mlp.pth"
LR_RESULTS_JSON = METRICS_DIR / "logistic_regression_results.json"
RF_RESULTS_JSON = METRICS_DIR / "random_forest_results.json"
MLP_RESULTS_JSON = METRICS_DIR / "mlp_results.json"

SHAP_SUMMARY_PNG = FIGURES_DIR / "shap_summary.png"
SHAP_BAR_PNG = FIGURES_DIR / "shap_bar.png"
SHAP_WATERFALL_PNG = FIGURES_DIR / "shap_waterfall.png"

MACCS_REFERENCE_XLSX = ROOT_DIR / "MACCS_Keys_Human_Readable_Reference(1).xlsx"


def ensure_standard_dirs() -> None:
    for path in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        ARTIFACTS_DIR,
        MODELS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        EXPLANATIONS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
