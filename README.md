# BIM Label Classifier — Results Dashboard

**View-only** Streamlit dashboard for sharing model evaluation results with stakeholders.  
All metrics and predictions are **pre-exported** in `bim_outputs/final/` — **no training runs in the app**.

> **Not LLM-based:** classical ML classifier (LightGBM), already trained on project data. The dashboard only displays results.

## For your manager — open the live page

Deploy from [share.streamlit.io](https://share.streamlit.io):

- **Repository:** `konskyrt/model-classification-app`
- **Main file:** `dashboard_updated.py`
- **Python:** 3.12 (via `.python-version`)

Send the app URL (e.g. `https://model-classification-app-xxxxx.streamlit.app`).

**If the build fails on Python 3.14:** delete the app on Streamlit Cloud and redeploy — pick **Python 3.12** under Advanced settings. The repo also includes `environment.yml` to force Python 3.12 via conda.

## Run locally (optional)

```powershell
pip install -r requirements.txt
streamlit run dashboard_updated.py
```

Open `http://localhost:8501`. Reads pre-computed files from `bim_outputs/final/` and shows label **names** alongside codes.

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

## Retrain the model (optional, not needed for the dashboard)

Training notebook and extra deps are kept for reference only (`bim_classifier_final.ipynb`, `requirements-train.txt`).

## Modeling approach

- **Target:** `Label Code`
- **Aggregation:** signature level (train/test split at signature level to avoid leakage)
- **Features:** TF-IDF text, one-hot categorical, scaled numerical geometry
- **Model:** LightGBM with class balancing

See `READMEmd.txt` for full setup and environment notes.
