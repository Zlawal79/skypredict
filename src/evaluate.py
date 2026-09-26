import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
)

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    FIGURE_DIR,
)


# =========================================================
# FILE LOCATIONS
# =========================================================

TEST_FILE = PROCESSED_DATA_DIR / "test_data.csv"

LOGISTIC_MODEL_FILE = (
    MODEL_DIR / "logistic_regression.joblib"
)

RANDOM_FOREST_MODEL_FILE = (
    MODEL_DIR / "random_forest.joblib"
)

RESULTS_FILE = (
    MODEL_DIR / "model_results.json"
)

TARGET = "DepDel15"


# =========================================================
# CREATE FIGURE FOLDER
# =========================================================

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# LOAD DATA AND MODELS
# =========================================================

print("\n====================================")
print("SKYPREDICT MODEL EVALUATION")
print("====================================")

print("\nLoading test dataset...")

test_data = pd.read_csv(TEST_FILE)

X_test = test_data.drop(
    columns=[TARGET]
)

y_test = test_data[TARGET]

print(
    f"Test rows: {len(test_data):,}"
)


print("\nLoading trained models...")

logistic_model = joblib.load(
    LOGISTIC_MODEL_FILE
)

random_forest_model = joblib.load(
    RANDOM_FOREST_MODEL_FILE
)


# =========================================================
# PREDICTIONS
# =========================================================

print("\nGenerating predictions...")

logistic_predictions = (
    logistic_model.predict(X_test)
)

random_forest_predictions = (
    random_forest_model.predict(X_test)
)

logistic_probabilities = (
    logistic_model.predict_proba(X_test)[:, 1]
)

random_forest_probabilities = (
    random_forest_model.predict_proba(X_test)[:, 1]
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n====================================")
print("RANDOM FOREST CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_test,
        random_forest_predictions,
        digits=4,
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print(
    "\nCreating confusion matrix..."
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    random_forest_predictions,
    display_labels=[
        "Not Delayed",
        "Delayed",
    ],
    ax=ax,
)

ax.set_title(
    "SkyPredict Random Forest Confusion Matrix"
)

plt.tight_layout()

confusion_file = (
    FIGURE_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    confusion_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# ROC CURVE
# =========================================================

print(
    "Creating ROC curve..."
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

RocCurveDisplay.from_predictions(
    y_test,
    logistic_probabilities,
    name="Logistic Regression",
    ax=ax,
)

RocCurveDisplay.from_predictions(
    y_test,
    random_forest_probabilities,
    name="Random Forest",
    ax=ax,
)

ax.set_title(
    "SkyPredict ROC Curve"
)

plt.tight_layout()

roc_file = (
    FIGURE_DIR
    / "roc_curve.png"
)

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# PRECISION-RECALL CURVE
# =========================================================

print(
    "Creating Precision-Recall curve..."
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

PrecisionRecallDisplay.from_predictions(
    y_test,
    logistic_probabilities,
    name="Logistic Regression",
    ax=ax,
)

PrecisionRecallDisplay.from_predictions(
    y_test,
    random_forest_probabilities,
    name="Random Forest",
    ax=ax,
)

ax.set_title(
    "SkyPredict Precision-Recall Curve"
)

plt.tight_layout()

pr_file = (
    FIGURE_DIR
    / "precision_recall_curve.png"
)

plt.savefig(
    pr_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# MODEL COMPARISON
# =========================================================

print(
    "Creating model comparison chart..."
)

with open(
    RESULTS_FILE,
    "r",
) as file:

    results = json.load(file)


metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

logistic_scores = [
    results["logistic_regression"][metric]
    for metric in metrics
]

forest_scores = [
    results["random_forest"][metric]
    for metric in metrics
]


comparison = pd.DataFrame(
    {
        "Logistic Regression":
            logistic_scores,

        "Random Forest":
            forest_scores,
    },
    index=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ],
)


ax = comparison.plot(
    kind="bar",
    figsize=(10, 6),
)

ax.set_title(
    "SkyPredict Model Performance Comparison"
)

ax.set_ylabel(
    "Score"
)

ax.set_ylim(
    0,
    1,
)

ax.set_xlabel(
    "Metric"
)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Model"
)

plt.tight_layout()

comparison_file = (
    FIGURE_DIR
    / "model_comparison.png"
)

plt.savefig(
    comparison_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

print(
    "Creating feature importance chart..."
)

preprocessor = (
    random_forest_model
    .named_steps["preprocessor"]
)

classifier = (
    random_forest_model
    .named_steps["classifier"]
)

feature_names = (
    preprocessor
    .get_feature_names_out()
)

importances = (
    classifier
    .feature_importances_
)


importance_data = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importances,
    }
)

importance_data = (
    importance_data
    .sort_values(
        "importance",
        ascending=False,
    )
)


# Save complete feature importance table
importance_csv = (
    FIGURE_DIR
    / "feature_importance.csv"
)

importance_data.to_csv(
    importance_csv,
    index=False,
)


# Plot top 15 features
top_features = (
    importance_data
    .head(15)
    .sort_values(
        "importance",
        ascending=True,
    )
)


fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.barh(
    top_features["feature"],
    top_features["importance"],
)

ax.set_title(
    "Top 15 SkyPredict Random Forest Features"
)

ax.set_xlabel(
    "Feature Importance"
)

plt.tight_layout()

importance_file = (
    FIGURE_DIR
    / "feature_importance.png"
)

plt.savefig(
    importance_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# COMPLETE
# =========================================================

print("\n====================================")
print("MODEL EVALUATION COMPLETE")
print("====================================")

print("\nFigures saved to:")

print(confusion_file)
print(roc_file)
print(pr_file)
print(comparison_file)
print(importance_file)

print("\nFeature importance data saved to:")

print(importance_csv)