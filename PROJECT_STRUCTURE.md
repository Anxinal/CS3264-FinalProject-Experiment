# Project Structure

This project now uses a standardized directory layout and centralized path mapping.

## Directory layout

```text
DDI-term-project/
├── artifacts/
│   ├── explanations/
│   │   └── trdp/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── data/
│   ├── processed/
│   └── raw/
├── pipelines/
│   ├── data/
│   ├── explainability/
│   └── models/
├── trdp/
├── project_paths.py
└── ... (source scripts)
```

## Path conventions

All major scripts use `project_paths.py` instead of ad-hoc relative paths.

- Processed arrays and sample files are written to `data/processed/`
- Trained models are written to `artifacts/models/`
- Metrics JSON files are written to `artifacts/metrics/`
- SHAP images are written to `artifacts/figures/`
- TRDP outputs are written to `artifacts/explanations/trdp/`

## Updated scripts

The following scripts now use standardized paths:

- `pipelines/data/encode.py`
- `pipelines/data/positive_construct.py`
- `pipelines/data/negative_construct.py`
- `pipelines/data/positive_add.py`
- `pipelines/data/negative_add.py`
- `pipelines/data/dataset_create.py`
- `pipelines/data/dataset_new_create.py`
- `pipelines/models/logistic_regression.py`
- `pipelines/models/random_forest.py`
- `pipelines/models/mlp.py`
- `pipelines/models/attention.py`
- `pipelines/models/full_attention.py`
- `pipelines/models/bi_full_attention.py`
- `pipelines/explainability/shap_analysis.py`
- `trdp/trdp_analysis.py`
- `trdp/trdp_chain_report.py`
- `trdp/trdp_conclusion.py`
- `trdp/trdp_pair_conclusion.py`
- `trdp/mechanism_hypothesis.py`

## Example commands

```bash
python pipelines/data/encode.py
python pipelines/data/positive_construct.py
python pipelines/data/negative_construct.py
python pipelines/data/dataset_create.py
python pipelines/models/random_forest.py
python pipelines/models/bi_full_attention.py
python pipelines/explainability/shap_analysis.py
```

## Notes

- Legacy output files are moved to `artifacts/legacy_root_outputs/` by maintenance scripts.
- You can safely re-run pipeline scripts; outputs will be generated in standardized folders.
