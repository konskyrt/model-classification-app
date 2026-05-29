
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import json

st.set_page_config(page_title="BIM Label Classifier Dashboard", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 1rem 1.1rem;
    border-radius: 14px;
}
section.main > div {
    padding-top: 0rem;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path("bim_outputs/final")
PRED_PATH = BASE_DIR / "predictions.csv"
METRICS_PATH = BASE_DIR / "metrics.csv"
CLASS_METRICS_PATH = BASE_DIR / "classification_report.csv"
SUMMARY_PATH = BASE_DIR / "model_summary.json"
LABEL_OBJECT_PATH = Path("label_object.json")
LABEL_NAMES_PATH = Path("label_names.json")


@st.cache_data
def load_label_names():
    names = {}
    if LABEL_NAMES_PATH.exists():
        with open(LABEL_NAMES_PATH, encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                names = loaded

    label_object = {}
    if LABEL_OBJECT_PATH.exists():
        with open(LABEL_OBJECT_PATH, encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                label_object = loaded

    return names, label_object


def lookup_label_name(code, names, label_object):
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return "Unknown"
    code = str(code).strip()
    if code in names:
        return names[code]
    if code in label_object:
        return label_object[code]
    parts = code.split(".")
    for i in range(len(parts), 0, -1):
        key = ".".join(parts[:i])
        if key in label_object:
            return label_object[key]
    return code


def format_label(code, names, label_object):
    name = lookup_label_name(code, names, label_object)
    return f"{code} — {name}" if name != code else code


@st.cache_data
def load_data():
    missing = [str(p) for p in [PRED_PATH, METRICS_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    pred_df = pd.read_csv(PRED_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)

    class_metrics_df = None
    if CLASS_METRICS_PATH.exists():
        class_metrics_df = pd.read_csv(CLASS_METRICS_PATH)
        if "label_code" not in class_metrics_df.columns:
            class_metrics_df = class_metrics_df.rename(columns={class_metrics_df.columns[0]: "label_code"})

    summary = {}
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                summary = loaded

    return pred_df, metrics_df, class_metrics_df, summary

def fmt_int_or_dash(value):
    return f"{int(value):,}" if pd.notnull(value) else "—"

def fmt_float_or_dash(value, decimals=1):
    return f"{float(value):.{decimals}f}" if pd.notnull(value) else "—"

def fmt_text_or_dash(value):
    if value is None:
        return "—"
    if isinstance(value, float) and pd.isna(value):
        return "—"
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return "—"
    return str(value)

try:
    pred_df, metrics_df, class_metrics_df, summary = load_data()
    label_names, label_object = load_label_names()
except Exception as e:
    st.error(f"Could not load dashboard inputs: {e}")
    st.stop()

label_name = lambda code: lookup_label_name(code, label_names, label_object)
label_display = lambda code: format_label(code, label_names, label_object)

metrics_row = metrics_df.iloc[0]

accuracy_val = summary.get("accuracy", metrics_row["accuracy"])
macro_f1_val = summary.get("macro_f1", metrics_row["macro_f1"])
weighted_f1_val = summary.get("weighted_f1", metrics_row["weighted_f1"])
test_signatures_val = summary.get("test_signatures", len(pred_df))
train_signatures_val = summary.get("train_signatures", np.nan)
num_classes_val = summary.get("num_classes", pred_df["true_label"].nunique())
mean_confidence_val = summary.get("mean_confidence", pred_df["confidence"].mean() if "confidence" in pred_df.columns else np.nan)

st.title("BIM Label Classifier — Evaluation Dashboard")
st.caption(
    "LightGBM classifier with signature-level evaluation, confidence analysis, per-class performance, "
    "confusion structure, and misclassification review."
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{accuracy_val*100:.2f}%")
c2.metric("Macro F1", f"{macro_f1_val:.4f}")
c3.metric("Weighted F1", f"{weighted_f1_val:.4f}")
c4.metric("Test Signatures", fmt_int_or_dash(test_signatures_val))
c5.metric("Train Signatures", fmt_int_or_dash(train_signatures_val))
c6.metric("Label Classes", fmt_int_or_dash(num_classes_val))

st.markdown("---")

st.subheader("Prediction Confidence Analysis")
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        pred_df,
        x="confidence",
        color="correct",
        nbins=40,
        barmode="overlay",
        color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
        title="Confidence Distribution (Correct vs Incorrect)",
        labels={"confidence": "Confidence Score", "count": "Count"}
    )
    fig.update_layout(height=420, legend_title_text="Prediction result", margin=dict(t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    pred_df["conf_bin"] = pd.cut(pred_df["confidence"], bins=10)
    conf_acc = pred_df.groupby("conf_bin", observed=True)["correct"].mean().reset_index()
    conf_acc["conf_bin"] = conf_acc["conf_bin"].astype(str)

    fig2 = px.bar(
        conf_acc,
        x="conf_bin",
        y="correct",
        title="Accuracy by Confidence Bucket",
        labels={"conf_bin": "Confidence Range", "correct": "Accuracy"},
        color="correct",
        color_continuous_scale="RdYlGn",
        text=conf_acc["correct"].map(lambda x: f"{x:.0%}")
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        height=420,
        margin=dict(t=60, b=110),
        xaxis_tickangle=-35,
        coloraxis_showscale=False
    )
    fig2.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

if class_metrics_df is not None and "label_code" in class_metrics_df.columns:
    st.subheader("Per-Class Performance")
    class_metrics = class_metrics_df[
        ~class_metrics_df["label_code"].isin(["accuracy", "macro avg", "weighted avg"])
    ].copy()
    class_metrics = class_metrics[pd.to_numeric(class_metrics["support"], errors="coerce") > 0]
    class_metrics["label_name"] = class_metrics["label_code"].map(label_name)
    class_metrics["label_display"] = class_metrics["label_code"].map(label_display)
    class_metrics = class_metrics.sort_values("f1-score", ascending=True)

    fig3 = px.bar(
        class_metrics,
        x="f1-score",
        y="label_display",
        orientation="h",
        color="f1-score",
        color_continuous_scale="RdYlGn",
        title="F1 Score per Label Class (Test Set)",
        labels={"f1-score": "F1 Score", "label_display": "Label"},
        height=max(900, len(class_metrics) * 28),
        text=class_metrics.apply(lambda r: f"{r['f1-score']:.2f} | n={int(r['support'])}", axis=1),
        hover_data={"label_code": True, "label_name": True, "label_display": False},
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=40, r=160, t=60, b=40),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    st.subheader("Precision vs Recall by Class")
    fig4 = px.scatter(
        class_metrics,
        x="precision",
        y="recall",
        size="support",
        color="f1-score",
        hover_name="label_display",
        custom_data=["label_code", "label_name"],
        color_continuous_scale="RdYlGn",
        title="Precision vs Recall (Bubble Size = Support)",
        labels={"precision": "Precision", "recall": "Recall"},
    )
    fig4.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Code: %{customdata[0]}<br>Name: %{customdata[1]}<br>Precision: %{x:.2f}<br>Recall: %{y:.2f}<extra></extra>"
    )
    fig4.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig4.update_layout(height=520, margin=dict(t=60, b=40))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

st.subheader("Confusion Matrix (Top 20 Classes by Support)")
top20 = pred_df["true_label"].value_counts().head(20).index.tolist()
top20_display = [label_display(code) for code in top20]
cm_top = pd.crosstab(
    pred_df["true_label"],
    pred_df["predicted_label"],
).reindex(index=top20, columns=top20, fill_value=0)
cm_top.index = top20_display
cm_top.columns = top20_display

fig5 = px.imshow(
    cm_top,
    color_continuous_scale="Blues",
    title="Confusion Matrix — Top 20 Classes",
    labels={"x": "Predicted", "y": "Actual", "color": "Count"},
    aspect="auto",
    height=820,
)
fig5.update_xaxes(tickangle=-45)
fig5.update_layout(margin=dict(t=60, b=180, l=220, r=30))
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

st.subheader("Sample Predictions with Confidence & Alternatives")
display_df = pred_df.copy().head(50)
for col in ["confidence", "alt_1_confidence", "alt_2_confidence"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")

for code_col, name_col in [
    ("true_label", "true_label_name"),
    ("predicted_label", "predicted_label_name"),
    ("alternative_1", "alternative_1_name"),
    ("alternative_2", "alternative_2_name"),
]:
    if code_col in display_df.columns:
        display_df[name_col] = display_df[code_col].map(label_name)

def highlight_correct(val):
    if val is True:
        return "background-color: rgba(46, 204, 113, 0.25)"
    if val is False:
        return "background-color: rgba(231, 76, 60, 0.25)"
    return ""

display_cols = [
    "signature",
    "true_label", "true_label_name",
    "predicted_label", "predicted_label_name",
    "confidence",
    "alternative_1", "alternative_1_name", "alt_1_confidence",
    "alternative_2", "alternative_2_name", "alt_2_confidence",
    "correct",
]
display_cols = [c for c in display_cols if c in display_df.columns]

st.dataframe(
    display_df[display_cols].style.applymap(
        highlight_correct,
        subset=["correct"] if "correct" in display_cols else []
    ),
    use_container_width=True,
    height=520
)

st.markdown("---")

st.subheader("Top Misclassified Label Pairs")
wrong = pred_df[pred_df["correct"] == False].copy()
if len(wrong) > 0:
    wrong["pair"] = wrong.apply(
        lambda r: f"{label_display(r['true_label'])} → {label_display(r['predicted_label'])}",
        axis=1,
    )
    top_wrong = wrong["pair"].value_counts().head(15).reset_index()
    top_wrong.columns = ["Misclassification Pair", "Count"]

    fig6 = px.bar(
        top_wrong,
        x="Count",
        y="Misclassification Pair",
        orientation="h",
        color="Count",
        color_continuous_scale="Reds",
        title="Most Common Misclassifications (True → Predicted)",
        text="Count"
    )
    fig6.update_traces(textposition="outside")
    fig6.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_range=[0, top_wrong["Count"].max() + 1],
        height=520,
        margin=dict(t=60, b=40, l=80, r=50),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("No misclassifications found in predictions.csv")

st.markdown("---")

with st.expander("Model & Training Details", expanded=True):
    col1, col2 = st.columns(2)

    feature_shape = summary.get("feature_matrix_shape", {}) or {}
    text_features = summary.get("text_features", {}) or {}
    word_tfidf = text_features.get("word_tfidf", {}) or {}
    char_tfidf = text_features.get("char_tfidf", {}) or {}
    categorical_features = summary.get("categorical_features", {}) or {}
    numerical_features = summary.get("numerical_features", {}) or {}

    feature_shape_text = "—"
    if pd.notnull(feature_shape.get("rows")) and pd.notnull(feature_shape.get("cols")):
        feature_shape_text = f"({int(feature_shape['rows']):,}, {int(feature_shape['cols']):,})"

    projects_covered = summary.get("projects_covered")
    if isinstance(projects_covered, list):
        projects_covered_text = ", ".join(projects_covered) if projects_covered else "—"
    else:
        projects_covered_text = "—"

    mean_conf_text = f"{float(mean_confidence_val):.2%}" if pd.notnull(mean_confidence_val) else "—"

    with col1:
        st.markdown(f"""
        **Model:** {fmt_text_or_dash(summary.get("model_name", "LightGBM"))}  
        **Experiment:** {fmt_text_or_dash(summary.get("experiment"))}  
        **n_estimators:** {fmt_text_or_dash(summary.get("n_estimators"))}  
        **learning_rate:** {fmt_text_or_dash(summary.get("learning_rate"))}  
        **num_leaves:** {fmt_text_or_dash(summary.get("num_leaves"))}  
        **min_child_samples:** {fmt_text_or_dash(summary.get("min_child_samples"))}  
        **reg_alpha:** {fmt_text_or_dash(summary.get("reg_alpha"))}  
        **reg_lambda:** {fmt_text_or_dash(summary.get("reg_lambda"))}  
        **Class balancing:** `{fmt_text_or_dash(summary.get("class_weight"))}`  
        **Train signatures:** {fmt_int_or_dash(train_signatures_val)}  
        **Test signatures:** {fmt_int_or_dash(test_signatures_val)}  
        **Total label classes:** {fmt_int_or_dash(num_classes_val)}  
        **Accuracy:** {accuracy_val*100:.2f}%  
        **Macro F1:** {macro_f1_val:.4f}  
        **Weighted F1:** {weighted_f1_val:.4f}  
        **Mean confidence:** {mean_conf_text}  
        **Training time:** {fmt_float_or_dash(summary.get("train_time_s"), 1)} seconds  
        """)

    with col2:
        st.markdown(f"""
        **Text features:** {fmt_text_or_dash(text_features.get("method"))}  
        **Word TF-IDF:** max {fmt_text_or_dash(word_tfidf.get("max_features"))}, ngrams={fmt_text_or_dash(tuple(word_tfidf.get("ngram_range", [])) if word_tfidf.get("ngram_range") else None)}, sublinear_tf={fmt_text_or_dash(word_tfidf.get("sublinear_tf"))}, min_df={fmt_text_or_dash(word_tfidf.get("min_df"))}  
        **Char TF-IDF:** max {fmt_text_or_dash(char_tfidf.get("max_features"))}, analyzer={fmt_text_or_dash(char_tfidf.get("analyzer"))}, ngrams={fmt_text_or_dash(tuple(char_tfidf.get("ngram_range", [])) if char_tfidf.get("ngram_range") else None)}, sublinear_tf={fmt_text_or_dash(char_tfidf.get("sublinear_tf"))}, min_df={fmt_text_or_dash(char_tfidf.get("min_df"))}  
        **Categorical:** {fmt_text_or_dash(categorical_features.get("encoder"))} (`{fmt_text_or_dash(", ".join(categorical_features.get("columns", [])) if categorical_features.get("columns") else None)}`)  
        **Numerical:** {fmt_text_or_dash(numerical_features.get("scaler"))} + {fmt_text_or_dash(numerical_features.get("imputation"))} imputation (`{fmt_text_or_dash(", ".join(numerical_features.get("columns", [])) if numerical_features.get("columns") else None)}`)  
        **Split strategy:** {fmt_text_or_dash(summary.get("split_strategy"))}  
        **Leakage prevention:** {fmt_text_or_dash(summary.get("leakage_prevention"))}  
        **Feature matrix shape:** {feature_shape_text}  
        **Projects covered:** {projects_covered_text}  
        **Output source:** `bim_outputs/final`  
        """)
