from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# =========================
# Constants (mirror the notebook)
# =========================
DATA_PATH = Path(__file__).parent / "vgsales.csv"

# Binary models: target is "Global_Sales above the dataset median".
# NA/EU/JP/Other sales are NOT used as features: they add up to Global_Sales,
# which defines the target, so using them would be target leakage.
NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = ["Publisher", "Year"]
CLASS_NAMES = ["At/below median", "Above median"]  # index = class (0 / 1)

# Three-class Random Forest used for the feature analysis tab
TIER_LABELS = ["Low", "Medium", "High"]
TIER_BINS = [-1, 1, 10, np.inf]  # Low <= 1M < Medium <= 10M < High
TIER_FEATURES = ["Platform", "Genre", "Publisher", "Year"]

# =========================
# Page configuration & styling
# =========================
st.set_page_config(page_title="Video Game Sales Prediction", layout="wide")

st.markdown(
    """
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem; border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    h1 { color: #667eea; text-align: center; font-weight: 700; margin-bottom: 0.5rem; }
    h2, h3 { color: #764ba2; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem;
        font-weight: 600; font-size: 16px; width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 8px 8px 0 0;
        padding: 10px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Video Game Sales Classification")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 18px;'>Predict whether a game's "
    "<strong>global sales will be above the dataset median</strong> from its publisher and "
    "release year</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

if not DATA_PATH.exists():
    st.error(f"Could not find `{DATA_PATH.name}`. Put it in the same folder as app.py.")
    st.stop()


# =========================
# Data
# =========================
@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Publisher"] = df["Publisher"].fillna("Unknown")
    df = df.dropna(subset=["Year", "Genre", "Platform", "Global_Sales"]).copy()
    df["Year"] = df["Year"].astype(int)
    return df.drop_duplicates()


# =========================
# Binary models (same pipelines as the notebook)
# =========================
def make_preprocess() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict:
    return {
        "Decision Tree": Pipeline(
            [("preprocess", make_preprocess()), ("model", DecisionTreeClassifier(random_state=42))]
        ),
        "Logistic Regression": Pipeline(
            [("preprocess", make_preprocess()), ("model", LogisticRegression(max_iter=2000))]
        ),
        "KNN": Pipeline(
            [("preprocess", make_preprocess()), ("model", KNeighborsClassifier())]
        ),
        "Random Forest": Pipeline(
            [("preprocess", make_preprocess()), ("model", RandomForestClassifier(random_state=42))]
        ),
        "ANN (MLP)": Pipeline(
            [
                ("preprocessor", make_preprocess()),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "ann",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate(model, X_test, y_test) -> tuple[dict, np.ndarray]:
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]  # class 1 = above median
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, proba),
    }
    return metrics, y_pred


@st.cache_resource(show_spinner="Training models (the ANN can take a little while)...")
def train_all():
    df = load_data()
    median = float(df["Global_Sales"].median())
    y = (df["Global_Sales"] > median).astype(int)
    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipelines, metrics, predictions = {}, {}, {}
    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
        metrics[name], predictions[name] = evaluate(pipe, X_test, y_test)

    majority = int(y_train.mode()[0])
    baseline_acc = float((y_test == majority).mean())
    return pipelines, metrics, predictions, y_test, median, majority, baseline_acc


# =========================
# Three-class Random Forest (feature analysis)
# =========================
@st.cache_resource(show_spinner="Training three-class Random Forest...")
def train_tier_model():
    df = load_data()
    y = pd.cut(df["Global_Sales"], bins=TIER_BINS, labels=TIER_LABELS).astype(str)
    X = df[TIER_FEATURES].copy()
    for col in ["Platform", "Genre", "Publisher"]:
        X[col] = LabelEncoder().fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    importance_df = (
        pd.DataFrame({"Feature": TIER_FEATURES, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T
    accuracy = accuracy_score(y_test, y_pred)
    baseline = float((y_test == y_train.mode()[0]).mean())
    return importance_df, report, accuracy, baseline


df = load_data()
pipelines, all_metrics, all_predictions, y_test, median_sales, majority_class, baseline_acc = train_all()

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Comparison", "Feature Analysis"])

# ---------- TAB 1: PREDICTION ----------
with tab1:
    model_name = st.selectbox("Select Machine Learning/Deep Learning Model", list(pipelines.keys()))
    model = pipelines[model_name]

    st.markdown("---")
    st.subheader("Enter Game Details")

    publisher_options = sorted(df["Publisher"].unique())
    year_options = sorted(df["Year"].unique())

    c1, c2 = st.columns(2)
    with c1:
        publisher = st.selectbox("Publisher", publisher_options)
    with c2:
        year = st.selectbox(
            "Release year",
            year_options,
            index=year_options.index(2010) if 2010 in year_options else 0,
        )

    st.caption(
        "Sales figures are not inputs: regional sales add up to the global sales the model is "
        "predicting, so a model that used them would just be doing arithmetic. "
        f"\"Above median\" means more than {median_sales:.2f} million units worldwide."
    )
    st.markdown("---")

    if st.button("Predict Sales Category"):
        input_df = pd.DataFrame({"Publisher": [publisher], "Year": [int(year)]})
        pred_class = int(model.predict(input_df)[0])
        probas = model.predict_proba(input_df)[0]  # order: class 0, class 1

        st.subheader("Prediction Results")
        result_col, chart_col = st.columns([1, 2])
        color_map = {0: "#ff6b6b", 1: "#4ecdc4"}
        pred_color = color_map[pred_class]

        with result_col:
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, {pred_color} 0%, {pred_color}dd 100%);
                            padding: 2rem; border-radius: 15px; text-align: center;
                            box-shadow: 0 8px 20px rgba(0,0,0,0.15);'>
                    <h2 style='color: white; margin: 0; font-size: 24px;'>Predicted Category</h2>
                    <h1 style='color: white; margin: 10px 0 0 0; font-size: 36px; font-weight: 800;'>{CLASS_NAMES[pred_class]}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Confidence levels:**")
            for idx, label in enumerate(CLASS_NAMES):
                st.markdown(f"**{label}:** {probas[idx] * 100:.1f}%")

        with chart_col:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(CLASS_NAMES, probas, color=[color_map[0], color_map[1]], height=0.6)
            for bar, prob in zip(bars, probas):
                ax.text(
                    bar.get_width() + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{prob * 100:.1f}%",
                    ha="left",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
            ax.set_xlabel("Probability", fontweight="bold")
            ax.set_ylabel("Sales category", fontweight="bold")
            ax.set_title("Prediction Probability Distribution", fontweight="bold", color="#667eea")
            ax.set_xlim(0, 1.15)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ---------- TAB 2: MODEL COMPARISON ----------
with tab2:
    st.subheader("Model Comparison")
    st.markdown("All models are evaluated on the same held-out 20% test split.")

    comparison_df = pd.DataFrame(all_metrics).T

    st.info(
        f"**Baseline:** always predicting \"{CLASS_NAMES[majority_class]}\" already gives "
        f"**{baseline_acc:.1%}** accuracy. Compare each model's accuracy against that number."
    )

    st.markdown("### Metrics summary")
    st.dataframe(
        comparison_df.style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=0),
    )

    st.markdown("### Best performing models")
    cols = st.columns(3)
    for metric, col in zip(["Accuracy", "F1-score", "AUC"], cols):
        best_model = comparison_df[metric].idxmax()
        best_score = comparison_df[metric].max()
        with col:
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 1.5rem; border-radius: 10px; text-align: center;'>
                    <h4 style='color: white; margin: 0; font-size: 16px;'>{metric}</h4>
                    <h3 style='color: white; margin: 5px 0; font-size: 20px;'>{best_model}</h3>
                    <p style='color: white; margin: 0; font-size: 24px; font-weight: bold;'>{best_score:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Visual comparison")
    v1, v2 = st.columns(2)

    with v1:
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
        ax.set_title("Model Performance Comparison", fontweight="bold", color="#764ba2")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.legend(title="Metrics", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with v2:
        radar_metrics = ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
        palette = ["#667eea", "#764ba2", "#ff6b6b", "#4ecdc4", "#f7b731"]
        for (name, m), color in zip(all_metrics.items(), palette):
            values = [m[k] for k in radar_metrics]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=name, color=color)
            ax.fill(angles, values, alpha=0.10, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics, size=9)
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Radar Chart", fontweight="bold", color="#764ba2", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("### Confusion matrices")
    items = list(all_predictions.items())
    for start in range(0, len(items), 3):
        row = st.columns(3)
        for col, (name, y_pred) in zip(row, items[start:start + 3]):
            with col:
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    cbar_kws={"label": "Count"},
                )
                ax.set_title(name, fontweight="bold", color="#764ba2")
                ax.set_xlabel("Predicted", fontweight="bold")
                ax.set_ylabel("Actual", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

# ---------- TAB 3: FEATURE ANALYSIS ----------
with tab3:
    importance_df, tier_report, tier_accuracy, tier_baseline = train_tier_model()

    st.subheader("Random Forest Feature Importance")
    st.markdown(
        "A separate Random Forest that classifies games into three tiers "
        f"(Low <= 1M, Medium <= 10M, High > 10M units) from Platform, Genre, Publisher and Year. "
        f"Test accuracy: **{tier_accuracy:.1%}** (always predicting the majority tier: "
        f"**{tier_baseline:.1%}**)."
    )

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(8, 5))
        ordered = importance_df.sort_values("Importance")
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ordered)))
        bars = ax.barh(ordered["Feature"], ordered["Importance"], color=colors)
        for bar in bars:
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {bar.get_width():.3f}", va="center", fontweight="bold")
        ax.set_title("Feature Importance Ranking", fontweight="bold", color="#764ba2")
        ax.set_xlabel("Importance")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            importance_df["Importance"],
            labels=importance_df["Feature"],
            autopct="%1.1f%%",
            colors=plt.cm.Set3(np.linspace(0, 1, len(importance_df))),
            startangle=90,
        )
        ax.set_title("Feature Importance Distribution", fontweight="bold", color="#764ba2")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    importance_df["Percentage"] = importance_df["Importance"] / importance_df["Importance"].sum() * 100
    importance_df["Cumulative %"] = importance_df["Percentage"].cumsum()
    st.dataframe(
        importance_df.style.format(
            {"Importance": "{:.6f}", "Percentage": "{:.2f}%", "Cumulative %": "{:.2f}%"}
        ).background_gradient(subset=["Importance"], cmap="YlGnBu"),
    )

    st.markdown("### Classification report (test set)")
    st.dataframe(tier_report.style.format("{:.3f}"))

    top = importance_df.iloc[0]
    i1, i2 = st.columns(2)
    with i1:
        st.info(f"**Most important feature:** {top['Feature']} ({top['Percentage']:.1f}% of total importance)")
    with i2:
        st.info(f"**Top 2 features:** {importance_df.head(2)['Percentage'].sum():.1f}% of total importance")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #999; font-size: 14px;'>Built with Streamlit & scikit-learn</p>",
    unsafe_allow_html=True,
)