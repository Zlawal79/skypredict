import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.config import PROCESSED_DATA_DIR, MODEL_DIR, RANDOM_STATE


# =========================================================
# FILE LOCATIONS
# =========================================================

DATA_FILE = PROCESSED_DATA_DIR / "model_data.csv"

LOGISTIC_MODEL_FILE = (
    MODEL_DIR / "logistic_regression.joblib"
)

RANDOM_FOREST_MODEL_FILE = (
    MODEL_DIR / "random_forest.joblib"
)

RESULTS_FILE = (
    MODEL_DIR / "model_results.json"
)


# =========================================================
# FEATURES
# =========================================================

TARGET = "DepDel15"


CATEGORICAL_FEATURES = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "time_period",
]


NUMERIC_FEATURES = [
    "DayOfWeek",
    "day_of_month",
    "departure_hour",
    "is_weekend",
    "CRSElapsedTime",
    "Distance",
    "temp",
    "dwpt",
    "rhum",
    "prcp",
    "prcp_reported",
    "wdir",
    "wspd",
    "coco",
]


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test,
):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    results = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    print(
        f"\n====================================\n"
        f"{model_name}\n"
        f"===================================="
    )

    for metric, value in results.items():

        print(
            f"{metric.upper():10s}: "
            f"{value:.4f}"
        )

    return results


# =========================================================
# TRAIN
# =========================================================

def train_models():

    print("\n====================================")
    print("SKYPREDICT MODEL TRAINING")
    print("====================================")

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    print("\nLoading model dataset...")

    data = pd.read_csv(DATA_FILE)

    print(
        f"Rows loaded: "
        f"{len(data):,}"
    )

    # -----------------------------------------------------
    # X AND Y
    # -----------------------------------------------------

    X = data[
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
    ]

    y = data[TARGET]

    print(
        f"Predictor features: "
        f"{X.shape[1]}"
    )

    # -----------------------------------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        f"\nTraining rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test):,}"
    )

    # -----------------------------------------------------
    # NUMERIC PREPROCESSING
    # -----------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # -----------------------------------------------------
    # CATEGORICAL PREPROCESSING
    # -----------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    # -----------------------------------------------------
    # COMBINED PREPROCESSOR
    # -----------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # -----------------------------------------------------
    # LOGISTIC REGRESSION
    # -----------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    logistic_results = evaluate_model(
        "LOGISTIC REGRESSION",
        logistic_model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    random_forest_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    random_forest_model.fit(
        X_train,
        y_train,
    )

    random_forest_results = evaluate_model(
        "RANDOM FOREST",
        random_forest_model,
        X_test,
        y_test,
    )

    # -----------------------------------------------------
    # SAVE MODELS
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        logistic_model,
        LOGISTIC_MODEL_FILE,
    )

    joblib.dump(
        random_forest_model,
        RANDOM_FOREST_MODEL_FILE,
    )

    # -----------------------------------------------------
    # SAVE TEST DATA
    # -----------------------------------------------------
    # We save the exact holdout set so evaluate.py can
    # generate confusion matrices and other figures later.

    X_test_output = X_test.copy()
    X_test_output[TARGET] = y_test.values

    test_file = (
        PROCESSED_DATA_DIR
        / "test_data.csv"
    )

    X_test_output.to_csv(
        test_file,
        index=False,
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    results = {
        "dataset_rows": len(data),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),

        "logistic_regression": {
            key: float(value)
            for key, value
            in logistic_results.items()
        },

        "random_forest": {
            key: float(value)
            for key, value
            in random_forest_results.items()
        },
    }

    with open(
        RESULTS_FILE,
        "w",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # FINISH
    # -----------------------------------------------------

    print("\n====================================")
    print("MODEL TRAINING COMPLETE")
    print("====================================")

    print("\nModels saved to:")

    print(LOGISTIC_MODEL_FILE)
    print(RANDOM_FOREST_MODEL_FILE)

    print("\nResults saved to:")

    print(RESULTS_FILE)

    print("\nTest dataset saved to:")

    print(test_file)


if __name__ == "__main__":

    train_models()