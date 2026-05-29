# BIM Taxonomy Classifier

Supervised **machine learning** workflow for predicting corrected BIM `Label Code` values from aggregated object-signature data.

> **Not LLM-based:** this is a classical ML classifier (LightGBM on engineered features — TF-IDF text, categorical encodings, geometry). It is trained on labeled project data and outputs confidence scores plus alternative classes. It does not use prompt-based or generative language models.

## Quick start — view results (dashboard)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run dashboard_updated.py
```

Open `http://localhost:8501`. The dashboard reads exported results from `bim_outputs/final/` and shows label **names** alongside codes (via `label_object.json` and `label_names.json`).

## Training datasets

Model trained on corrected BIM exports from six projects:

- HIAG
- HHM
- ELP
- BB07
- Burckhardt
- Spiez

~395k BIM objects aggregated to **3,119 signatures** (2,495 train / 624 test), **97 label classes**.

## Final model results

| Metric | Value |
|---|---|
| Accuracy | 88.5% |
| Macro F1 | 0.64 |
| Weighted F1 | 0.89 |
| Mean confidence | 91.8% |

See `experiment_comparison_metrics.png` for experiment comparison and `bim_outputs/final/` for exported metrics and predictions.

## Repository layout

```
├── bim_classifier_final.ipynb      # Final training notebook
├── dashboard_updated.py            # Streamlit evaluation dashboard
├── requirements.txt
├── label_object.json               # Label code → name taxonomy
├── label_names.json                # Model-specific label name lookup
├── experiment_comparison_metrics.png
├── bim_outputs/final/              # Final model outputs (metrics, predictions)
└── baseline model files/           # Baseline comparison artifacts
```

## Retrain the model

1. Place corrected source Excel files in your data directory (see `READMEmd.txt` for expected filenames).
2. Update `DATA_DIR` in `bim_classifier_final.ipynb`.
3. Run the notebook top to bottom — outputs are written to `bim_outputs/final/`.

## Modeling approach

- **Target:** `Label Code`
- **Aggregation:** signature level (train/test split at signature level to avoid leakage)
- **Features:** TF-IDF text, one-hot categorical, scaled numerical geometry
- **Model:** LightGBM with class balancing

See `READMEmd.txt` for full setup and environment notes.
