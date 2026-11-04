import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

st.set_page_config(page_title="Landslide Susceptibility Classification", page_icon="🌋", layout="wide")

SMOTE_EXTRA_SAMPLES = 1200
DATASET_PATH = Path(__file__).with_name("landslide_dataset.csv")

@st.cache_data
def load_data():
    data = pd.read_csv(DATASET_PATH)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["Landslide"]).drop_duplicates().reset_index(drop=True)
    data["Landslide"] = data["Landslide"].astype(int)
    return data

def smote_strategy(y):
    """Request exactly 1,200 synthetic rows for the minority class."""
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) != 2:
        raise ValueError("SMOTE requires exactly two target classes.")

    minority_class = class_counts.idxmin()
    return {minority_class: int(class_counts[minority_class]) + SMOTE_EXTRA_SAMPLES}

@st.cache_resource
def train_models(df):
    X = df.drop("Landslide", axis=1)
    y = df["Landslide"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    models = {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(
                sampling_strategy=smote_strategy, random_state=42
            )),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=5,
                max_features="sqrt", random_state=42, class_weight="balanced"
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(
                sampling_strategy=smote_strategy, random_state=42
            )),
            ("scaler", RobustScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=1000, random_state=42)),
        ]),
    }

    results, predictions, probabilities = [], {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="accuracy", n_jobs=-1
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        predictions[name], probabilities[name] = pred, prob
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
            "CV Accuracy Mean": cv_scores.mean(),
            "CV Accuracy Std": cv_scores.std(),
        })
    return (
        models, X_test, y_test, predictions, probabilities, pd.DataFrame(results),
        len(X_train), len(X_train) + SMOTE_EXTRA_SAMPLES, len(X_test)
    )

def compare_models_statistically(y_true, baseline_predictions, candidate_predictions,
                                 n_bootstrap=5000, n_permutations=10000, random_state=42):
    """Estimate the candidate-baseline accuracy difference and test paired predictions."""
    y_true = np.asarray(y_true)
    baseline_correct = (np.asarray(baseline_predictions) == y_true).astype(float)
    candidate_correct = (np.asarray(candidate_predictions) == y_true).astype(float)
    observed_difference = float(np.mean(candidate_correct - baseline_correct))

    rng = np.random.default_rng(random_state)
    paired_differences = candidate_correct - baseline_correct
    bootstrap_indices = rng.integers(0, len(y_true), size=(n_bootstrap, len(y_true)))
    bootstrap_differences = paired_differences[bootstrap_indices].mean(axis=1)
    confidence_interval = np.percentile(bootstrap_differences, [2.5, 97.5])

    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(y_true)))
    permuted_differences = (paired_differences * signs).mean(axis=1)
    p_value = float(
        (np.count_nonzero(np.abs(permuted_differences) >= abs(observed_difference)) + 1)
        / (n_permutations + 1)
    )
    return observed_difference, confidence_interval, p_value

def bootstrap_metric_interval(y_true, predictions, metric, n_bootstrap=5000, random_state=42):
    """Estimate a percentile confidence interval for a test-set metric."""
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(y_true), size=(n_bootstrap, len(y_true)))
    scores = np.array([
        metric(y_true[index], predictions[index])
        for index in indices
    ])
    return np.percentile(scores, [2.5, 97.5])

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Dataset not found: {DATASET_PATH}")
    st.stop()

(
    models, X_test, y_test, predictions, probabilities, results,
    original_train_size, augmented_train_size, test_size
) = train_models(df)

st.title("🌋 Landslide Susceptibility Classification")
st.write("Machine learning classification using Random Forest and Logistic Regression.")
st.caption(
    "This demonstration uses a reproducible, noisy simulated dataset. It is not a "
    "validated geological field dataset."
)

page = st.sidebar.radio("Navigation", ["Dashboard", "Model Comparison", "Statistical Evaluation", "Confusion Matrices", "ROC Curves", "Feature Importance", "Predict Susceptibility", "Dataset"])

if page == "Dashboard":
    st.header("📊 Project Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", len(df)); c2.metric("Features", df.shape[1]-1)
    c3.metric("Landslide Samples", int(df["Landslide"].sum())); c4.metric("Non-Landslide Samples", int((df["Landslide"] == 0).sum()))
    st.info(
        f"Original training rows: {original_train_size} | "
        f"Augmented training rows: {augmented_train_size} | "
        f"Untouched test rows: {test_size}"
    )
    st.subheader("Dataset Preview"); st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(7,4)); sns.countplot(x="Landslide", data=df, ax=ax); ax.set_title("Landslide Class Distribution"); st.pyplot(fig); plt.close(fig)
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(11,7)); sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax); ax.set_title("Correlation Matrix"); st.pyplot(fig); plt.close(fig)
    st.subheader("Data Quality Summary")
    quality = pd.DataFrame({
        "Missing Values": df.isna().sum(),
        "Missing (%)": (df.isna().mean() * 100).round(2),
        "Unique Values": df.nunique(),
    })
    st.dataframe(quality, use_container_width=True)

elif page == "Model Comparison":
    st.header("🤖 Model Performance Comparison")
    if results["Accuracy"].eq(1.0).all():
        st.warning(
            "Both models achieve perfect accuracy on this dataset. This indicates that the "
            "classes are perfectly separable, so the current data cannot demonstrate a "
            "meaningful performance difference."
        )
    st.dataframe(results.style.format({c:"{:.4f}" for c in results.columns if c != "Model"}), use_container_width=True)
    fig, ax = plt.subplots(figsize=(11,6)); results.set_index("Model").plot(kind="bar", ax=ax); ax.set_ylim(0,1.05); ax.set_ylabel("Score"); ax.set_title("Model Performance Comparison"); ax.tick_params(axis="x", rotation=0); st.pyplot(fig); plt.close(fig)
    selected = st.selectbox("Classification report", list(models.keys()))
    report = classification_report(y_test, predictions[selected], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"), use_container_width=True)
    accuracy_interval = bootstrap_metric_interval(
        y_test, predictions[selected], accuracy_score
    )
    st.caption(
        f"Bootstrap 95% accuracy interval for {selected}: "
        f"[{accuracy_interval[0]:.4f}, {accuracy_interval[1]:.4f}]"
    )

elif page == "Statistical Evaluation":
    st.header("📐 Statistical Evaluation")
    st.write("The two models are evaluated on the same held-out test set.")
    observed_difference, confidence_interval, p_value = compare_models_statistically(
        y_test, predictions["Logistic Regression"], predictions["Random Forest"]
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest - Logistic Accuracy", f"{observed_difference:.4f}")
    c2.metric("95% Bootstrap CI", f"[{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
    c3.metric("Paired permutation p-value", f"{p_value:.4f}")
    if results["Accuracy"].eq(1.0).all():
        st.warning(
            "Both models classify every test row correctly. The zero difference is a "
            "property of this dataset, not evidence that the models are equivalent on "
            "other landslide data."
        )
    st.markdown(
        "**Null hypothesis:** the two models have equal accuracy on the population represented "
        "by the test set. The bootstrap interval estimates the uncertainty in the observed "
        "accuracy difference, while the paired permutation test uses each sample's paired "
        "correct/incorrect outcomes."
    )
    if confidence_interval[0] > 0 and p_value < 0.05:
        st.success("The Random Forest improvement is statistically supported at the 5% level.")
    elif p_value < 0.05:
        st.warning("The models differ significantly, but the interval does not conclusively support a Random Forest improvement.")
    else:
        st.info("The observed difference is not statistically significant at the 5% level.")

elif page == "Confusion Matrices":
    st.header("🔲 Confusion Matrix")
    selected = st.selectbox("Select model", list(models.keys()))
    cm = confusion_matrix(y_test, predictions[selected])
    fig, ax = plt.subplots(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Landslide","Landslide"], yticklabels=["No Landslide","Landslide"], ax=ax); ax.set_title(f"{selected} Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig); plt.close(fig)

elif page == "ROC Curves":
    st.header("📈 ROC Curve Comparison")
    fig, ax = plt.subplots(figsize=(9,6))
    for name in models:
        fpr, tpr, _ = roc_curve(y_test, probabilities[name]); auc = roc_auc_score(y_test, probabilities[name]); ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0,1],[0,1],"--",label="Random Classifier"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve Comparison"); ax.legend(); ax.grid(alpha=.25); st.pyplot(fig); plt.close(fig)

elif page == "Feature Importance":
    st.header("🌳 Random Forest Feature Importance")
    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": models["Random Forest"].named_steps["model"].feature_importances_
    }).sort_values("Importance", ascending=False)
    st.dataframe(importance.style.format({"Importance":"{:.4f}"}), use_container_width=True)
    fig, ax = plt.subplots(figsize=(10,6)); sns.barplot(x="Importance", y="Feature", data=importance, ax=ax); ax.set_title("Random Forest Feature Importance"); st.pyplot(fig); plt.close(fig)

elif page == "Predict Susceptibility":
    st.header("🔮 Predict Landslide Susceptibility")
    selected = st.selectbox("Prediction model", list(models.keys()))
    c1,c2 = st.columns(2)
    with c1:
        rainfall=st.number_input("Rainfall (mm)", min_value=0.0, value=100.0, step=1.0)
        slope=st.number_input("Slope Angle", min_value=0.0, value=30.0, step=1.0)
        saturation=st.number_input("Soil Saturation", min_value=0.0, value=0.50, step=0.01)
        vegetation=st.number_input("Vegetation Cover", min_value=0.0, value=0.50, step=0.01)
    with c2:
        earthquake=st.number_input("Earthquake Activity", min_value=0.0, value=0.20, step=0.01)
        water=st.number_input("Proximity to Water", min_value=0.0, value=0.50, step=0.01)
        soil=st.selectbox("Soil Type", ["Gravel","Sand","Silt"])
    data = pd.DataFrame([{
        "Rainfall_mm":rainfall,"Slope_Angle":slope,"Soil_Saturation":saturation,"Vegetation_Cover":vegetation,
        "Earthquake_Activity":earthquake,"Proximity_to_Water":water,
        "Soil_Type_Gravel":int(soil=="Gravel"),"Soil_Type_Sand":int(soil=="Sand"),"Soil_Type_Silt":int(soil=="Silt")
    }])
    if st.button("🚀 Predict Landslide Susceptibility", use_container_width=True):
        model=models[selected]; pred=model.predict(data)[0]; prob=model.predict_proba(data)[0,1]
        if pred==1: st.error("⚠️ Predicted Class: LANDSLIDE")
        else: st.success("✅ Predicted Class: NO LANDSLIDE")
        st.metric("Landslide Probability", f"{prob*100:.2f}%")
        level="Low" if prob<.33 else "Moderate" if prob<.66 else "High"
        st.subheader("Susceptibility Level"); st.progress(float(prob)); st.write(f"### {level} Susceptibility")
        st.subheader("Input Values"); st.dataframe(data, use_container_width=True)
        st.caption("Low/Moderate/High thresholds are illustrative probability categories, not official geological hazard levels.")

elif page == "Dataset":
    st.header("📁 Dataset Information")
    st.write("Dataset shape:", df.shape)
    info=pd.DataFrame({"Column":df.columns,"Data Type":df.dtypes.astype(str).values,"Missing Values":df.isnull().sum().values,"Unique Values":[df[c].nunique() for c in df.columns]})
    st.subheader("Column Information"); st.dataframe(info,use_container_width=True)
    numeric_features = df.drop(columns=["Landslide"]).select_dtypes(include=np.number)
    quartiles = numeric_features.quantile([0.25, 0.75])
    iqr = quartiles.loc[0.75] - quartiles.loc[0.25]
    outlier_counts = (
        (numeric_features.lt(quartiles.loc[0.25] - 1.5 * iqr)
         | numeric_features.gt(quartiles.loc[0.75] + 1.5 * iqr))
        .sum()
        .rename("IQR Outlier Count")
        .to_frame()
    )
    st.subheader("IQR Outlier Screening")
    st.dataframe(outlier_counts, use_container_width=True)
    st.subheader("Descriptive Statistics"); st.dataframe(df.describe().T,use_container_width=True)
    st.subheader("Duplicate Rows"); st.write(int(df.duplicated().sum()))
    st.subheader("Full Dataset"); st.dataframe(df,use_container_width=True)

st.sidebar.divider(); st.sidebar.caption("Random Forest • Logistic Regression")
