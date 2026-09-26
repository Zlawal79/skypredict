import pandas as pd

from src.config import MERGED_FILE, PROCESSED_DATA_DIR


OUTPUT_FILE = PROCESSED_DATA_DIR / "model_data.csv"


def create_features():

    print("\n====================================")
    print("SKYPREDICT FEATURE ENGINEERING")
    print("====================================")

    # -----------------------------------------------------
    # LOAD MERGED DATA
    # -----------------------------------------------------

    data = pd.read_csv(
        MERGED_FILE,
        parse_dates=[
            "FlightDate",
            "scheduled_departure",
            "weather_hour",
        ],
    )

    print(
        f"\nMerged rows loaded: "
        f"{len(data):,}"
    )

    # -----------------------------------------------------
    # TIME FEATURES
    # -----------------------------------------------------

    data["departure_hour"] = (
        data["scheduled_departure"].dt.hour
    )

    data["departure_month"] = (
        data["scheduled_departure"].dt.month
    )

    data["day_of_month"] = (
        data["scheduled_departure"].dt.day
    )

    data["is_weekend"] = (
        data["DayOfWeek"]
        .isin([6, 7])
        .astype(int)
    )

    # -----------------------------------------------------
    # TIME-OF-DAY CATEGORY
    # -----------------------------------------------------

    def get_time_period(hour):

        if 5 <= hour < 12:
            return "morning"

        elif 12 <= hour < 17:
            return "afternoon"

        elif 17 <= hour < 22:
            return "evening"

        else:
            return "night"

    data["time_period"] = (
        data["departure_hour"]
        .apply(get_time_period)
    )

    # -----------------------------------------------------
    # WEATHER FEATURES
    # -----------------------------------------------------
    # prcp has many missing observations, so we keep the
    # raw column but also create an indicator showing whether
    # precipitation was actually reported.

    data["prcp_reported"] = (
        data["prcp"]
        .notna()
        .astype(int)
    )

    # -----------------------------------------------------
    # SELECT MODEL FEATURES
    # -----------------------------------------------------
    #
    # IMPORTANT:
    #
    # We intentionally exclude:
    #
    # DepDelayMinutes
    # CarrierDelay
    # WeatherDelay
    # NASDelay
    # SecurityDelay
    # LateAircraftDelay
    #
    # because these contain information about the outcome
    # and would cause target leakage.
    # -----------------------------------------------------

    model_columns = [

        # Target
        "DepDel15",

        # Airline / route
        "Reporting_Airline",
        "Origin",
        "Dest",

        # Calendar / scheduled time
        "DayOfWeek",
        "day_of_month",
        "departure_hour",
        "time_period",
        "is_weekend",

        # Scheduled flight information
        "CRSElapsedTime",
        "Distance",

        # Weather
        "temp",
        "dwpt",
        "rhum",
        "prcp",
        "prcp_reported",
        "wdir",
        "wspd",
        "coco",
    ]

    model_data = data[
        model_columns
    ].copy()

    # -----------------------------------------------------
    # CLEAN NUMERIC FEATURES
    # -----------------------------------------------------

    numeric_columns = [
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

    for column in numeric_columns:

        model_data[column] = pd.to_numeric(
            model_data[column],
            errors="coerce",
        )

    # -----------------------------------------------------
    # CHECK MISSING VALUES
    # -----------------------------------------------------

    print("\nMissing values before model preprocessing:")

    print(
        model_data
        .isna()
        .sum()
    )

    # -----------------------------------------------------
    # TARGET CHECK
    # -----------------------------------------------------

    model_data = model_data.dropna(
        subset=["DepDel15"]
    )

    model_data["DepDel15"] = (
        model_data["DepDel15"]
        .astype(int)
    )

    # -----------------------------------------------------
    # SAVE MODEL DATA
    # -----------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n====================================")
    print("FEATURE ENGINEERING COMPLETE")
    print("====================================")

    print(
        f"\nModel rows: "
        f"{len(model_data):,}"
    )

    print(
        f"Model features: "
        f"{len(model_data.columns) - 1}"
    )

    print("\nModel columns:")

    for column in model_data.columns:
        print(f" - {column}")

    print("\nTarget distribution:")

    print(
        model_data["DepDel15"]
        .value_counts()
        .sort_index()
    )

    print("\nTarget percentages:")

    print(
        model_data["DepDel15"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nModel dataset saved to:")

    print(OUTPUT_FILE)


if __name__ == "__main__":

    create_features()