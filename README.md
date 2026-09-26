# SkyPredict

**Machine Learning Flight Delay Prediction Using Flight and Weather Data**

SkyPredict is an end-to-end machine learning project that investigates whether a flight's departure delay can be predicted using scheduled flight information and historical weather conditions.

The project combines real-world U.S. flight data with hourly weather observations, builds a reproducible data-processing and machine-learning pipeline, compares multiple classification models, and presents the results through an interactive Streamlit application.

---

## Project Motivation

Flight delays affect passengers, airlines, airports, and transportation networks. While delays can happen for many reasons, information available before departure—such as scheduled flight characteristics, time of day, airport, airline, and weather conditions—may contain useful predictive signals.

SkyPredict explores the question:

> **Can we predict whether a flight will be delayed using flight information and weather conditions?**

The goal is not to create a production airline forecasting system, but to demonstrate an interpretable, evidence-based machine learning workflow using real public transportation and weather data.

---

## How SkyPredict Works

The project follows an end-to-end machine learning workflow:

1. Collect real historical flight data.
2. Collect hourly weather observations for selected airports.
3. Clean and standardize both datasets.
4. Match flights with weather conditions by airport and time.
5. Engineer predictive features.
6. Train multiple classification models.
7. Evaluate the models on held-out test data.
8. Analyze model behaviour and feature importance.
9. Present the workflow and results through an interactive Streamlit interface.

---

## Data

### Flight Data

Flight records come from the **U.S. Bureau of Transportation Statistics (BTS) Reporting Carrier On-Time Performance dataset**.

The project uses **July 2026** flight records.

- Raw flight records: **631,970**
- Cleaned flight records: **612,038**

The prediction target is based on `DepDel15`:

- `0` — flight departed less than 15 minutes late
- `1` — flight departed 15 minutes or more late

### Weather Data

Historical hourly weather observations were collected for **20 airports**.

- Weather observations: **14,880**
- Flights from weather-covered origins: **308,471**
- Successfully matched flight-weather records: **308,396**
- Weather matching rate: **99.98%**

Large raw and processed datasets are intentionally excluded from this repository.

---

## Feature Engineering

SkyPredict transforms raw flight and weather information into features suitable for machine learning.

Examples include:

- Scheduled departure hour
- Time of day
- Day of week
- Airline
- Origin airport
- Destination airport
- Temperature
- Relative humidity
- Precipitation
- Wind conditions
- Weather condition codes

Features that would leak information about the actual outcome—such as actual departure information or recorded delay causes—are not used as predictors.

---

## Machine Learning Models

Two classification models were evaluated:

### Logistic Regression

Used as an interpretable baseline model.

| Metric | Result |
|---|---:|
| Accuracy | 65.96% |
| Precision | 47.11% |
| Recall | 66.29% |
| F1 Score | 0.5508 |
| ROC-AUC | 0.7157 |

### Random Forest

Used to capture nonlinear relationships and interactions between flight and weather variables.

| Metric | Result |
|---|---:|
| Accuracy | 68.09% |
| Precision | 49.49% |
| Recall | 67.11% |
| F1 Score | 0.5697 |
| ROC-AUC | **0.7427** |

The Random Forest produced stronger results than the Logistic Regression baseline across the reported held-out evaluation metrics.

Because approximately **68.52%** of the merged flights were not delayed, accuracy alone is not an appropriate measure of model quality. The project therefore emphasizes metrics such as **ROC-AUC, recall, precision, and F1 score**.

---

## Model Evaluation

The repository includes several evaluation artifacts:

- Confusion matrix
- ROC curve
- Precision-recall curve
- Model comparison
- Random Forest feature importance

These visualizations are available in:

`reports/figures/`

---

## Model Explainability

Random Forest feature importance is used to examine which variables contributed most strongly to the model's predictions.

Some of the strongest features included:

- Departure hour
- Morning departure indicator
- Evening departure indicator
- Temperature
- Weather condition
- Relative humidity

Feature importance describes how useful a feature was to the trained model. It should **not** be interpreted as evidence that the feature directly causes flight delays.

---

## Interactive Application

SkyPredict includes a Streamlit interface designed to make the machine learning workflow understandable to both technical and non-technical users.

The application contains three main sections:

### About & How It Works

Introduces the problem, motivation, data pipeline, and machine learning workflow.

### Try It Out

Allows users to enter a hypothetical flight scenario and explore the model's estimated delay probability.

### Results & Insights

Presents model performance, evaluation visualizations, and key findings from the analysis.

---

## Project Structure

```text
SkyPredict/
│
├── app/
│   └── app.py
│
├── models/
│   ├── logistic_regression.joblib
│   └── model_results.json
│
├── reports/
│   └── figures/
│
├── src/
│   ├── clean_flights.py
│   ├── clean_weather.py
│   ├── config.py
│   ├── download_weather.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── load_data.py
│   ├── merge_datasets.py
│   ├── predict.py
│   └── train.py
│
├── .gitignore
├── requirements.txt
└── README.md