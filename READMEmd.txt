# BIM Label Classification Deliverable

This deliverable contains the ML classification workflow for predicting corrected BIM `Label Code` values from aggregated object-signature data.

## Included Files

- `bim_classifier_final.ipynb` — final model notebook containing the final selected model code only
- `dashboard_updated.py` — Streamlit dashboard for evaluation results
- `experiment_comparison_metrics.png` — experiment comparison visualization
- `requirements.txt` — Python dependencies
- `READMEmd.txt` — project documentation in text format
- `baseline model files/` — separate folder containing baseline model results and supporting evaluation files
- `bim_outputs/final/` — final exported predictions, metrics, and model summary files

## Deliverables Folder Structure

bim-label-classifier-deliverable/
├── baseline model files/
│   ├── baseline_model_evaluation_metrics.png
│   ├── confusion_matrix.csv
│   ├── model_summary.json
│   ├── per_class_metrics.csv
│   └── predictions_with_confidence.csv
├── bim_classifier_final.ipynb
├── bim_outputs/
│   └── final/
│       ├── classification_report.csv
│       ├── metrics.csv
│       ├── metrics.json
│       ├── model_summary.json
│       └── predictions.csv
├── dashboard_updated.py
├── experiment_comparison_metrics.png
├── READMEmd.txt
└── requirements.txt

## Problem Scope

The solution:

- loads corrected BIM classified datasets
- normalizes varying column names across source files
- aggregates rows at signature level
- avoids leakage by splitting train/test at signature level
- trains a supervised model to predict `Label Code`
- uses text, categorical, and numerical attributes as features
- outputs predictions, confidence scores, and top alternative classes
- provides evaluation results in dashboard format

## Environment Setup

Option 1 — Conda environment

Create and activate the environment:

conda create -n bim-label-classifier python=3.11 -y
conda activate bim-label-classifier
pip install -r requirements.txt

If you want to register the environment as a Jupyter kernel:

python -m ipykernel install --user --name bim-label-classifier --display-name "Python (bim-label-classifier)"

Option 2 — venv

For macOS/Linux:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

For Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Expected Input Files

The notebook expects the corrected BIM source files to be available in the configured data directory.

Expected source files:

- `HIAG_unified_corrected_labels.xlsx`
- `HHM_unified_corrected_labels.xlsx`
- `ELP_unified_corrected_labels (1).xlsx`
- `Burckhardt_unified_corrected_labels.xlsx`
- `BB07_unified_corrected_labels.xlsx`
- `Spiez_BIM_data_export_unified_corrected_labels.xlsx`

Update the dataset folder path inside the notebook if needed. The current pattern is:

DATA_DIR = Path.home() / "Downloads"

## Notebook Scope

- `bim_classifier_final.ipynb` contains the final selected model code only.
- Baseline model outputs and supporting evaluation files are stored separately in the `baseline model files/` folder.
- This keeps the final notebook focused on the production-ready workflow while preserving baseline comparison artifacts separately.

## Run the Notebook

Start Jupyter:

jupyter lab

Then open:

bim_classifier_final.ipynb

Run the notebook top to bottom to:

1. load and normalize the source files
2. filter corrected/passed labeled rows
3. aggregate at signature level
4. build features
5. train the final classifier
6. export predictions and evaluation files
7. write final output files to `bim_outputs/final`

## Run the Dashboard

Start the Streamlit dashboard with:

streamlit run dashboard_updated.py

The dashboard reads final output files from the exported output location used by the notebook.

## Modeling Notes

- Target variable: `Label Code`
- Aggregation level: signature
- Text features: TF-IDF on aggregated text fields
- Categorical features: one-hot encoding
- Numerical features: imputation + scaling
- Final selected model: LightGBM
- Evaluation includes confidence analysis, per-class metrics, confusion structure, and top misclassifications

## Final Output Files

The `bim_outputs/final/` folder contains:

- `classification_report.csv` — per-class evaluation report
- `metrics.csv` — summary metrics in tabular format
- `metrics.json` — summary metrics in JSON format
- `model_summary.json` — model metadata and summary information
- `predictions.csv` — predictions with confidence and alternative class outputs

## Baseline Folder Contents

The `baseline model files/` folder contains:

- `baseline_model_evaluation_metrics.png`
- `confusion_matrix.csv`
- `model_summary.json`
- `per_class_metrics.csv`
- `predictions_with_confidence.csv`

These files are retained separately for baseline reference and comparison.

## Notes

- Train/test splitting is done at signature level to avoid leakage.
- Column normalization is included to handle inconsistent source column names.
- The final notebook contains only the final selected model workflow.
- Baseline results and supporting files are kept in a separate folder for clarity.
- The dashboard provides a reporting layer on top of the exported final outputs.
