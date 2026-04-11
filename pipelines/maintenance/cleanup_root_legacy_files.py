from pathlib import Path
import shutil


ROOT_DIR = Path(__file__).resolve().parents[2]
LEGACY_DIR = ROOT_DIR / "artifacts" / "legacy_root_outputs"

LEGACY_FILES = [
    "X_train.npy",
    "X_val.npy",
    "X_test.npy",
    "y_train.npy",
    "y_val.npy",
    "y_test.npy",
    "positive_samples.npz",
    "negative_samples.npz",
    "positive_add_samples.npz",
    "negative_add_samples.npz",
    "drug_fingerprints.npz",
    "lr_model.pkl",
    "rf_model.pkl",
    "mlp.pth",
    "mlp_attn.pth",
    "full_mlp_attn.pth",
    "bi_mlp_attn.pth",
    "logistic_regression_results.json",
    "random_forest_results.json",
    "mlp_results.json",
    "attention_results.json",
    "full_attention_results.json",
    "bi_attention_results.json",
    "shap_summary.png",
    "shap_bar.png",
    "shap_waterfall.png",
]


def main():
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    moved = []
    skipped = []
    for name in LEGACY_FILES:
        src = ROOT_DIR / name
        dst = LEGACY_DIR / name
        if not src.exists():
            skipped.append(name)
            continue
        if dst.exists():
            src.unlink()
            moved.append(f"{name} (deleted duplicate)")
            continue
        shutil.move(str(src), str(dst))
        moved.append(name)

    print(f"Moved/de-duplicated: {len(moved)}")
    for item in moved:
        print(f"- {item}")
    print(f"Skipped(not found): {len(skipped)}")


if __name__ == "__main__":
    main()
