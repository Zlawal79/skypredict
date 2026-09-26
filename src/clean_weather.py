import pandas as pd

from src.config import WEATHER_FILE, PROCESSED_DATA_DIR


def clean_weather_data(file_path=WEATHER_FILE):
    """
    Clean the raw Meteostat hourly weather dataset
    for the SkyPredict machine-learning pipeline.
    """

    print("\n====================================")
    print("CLEANING WEATHER DATA")
    print("====================================")

    # Load raw weather data
    weather = pd.read_csv(file_path)

    print(f"\nRaw weather rows: {len(weather):,}")

    # -----------------------------------------------------
    # KEEP USEFUL WEATHER FEATURES
    # -----------------------------------------------------

    columns_to_keep = [
        "time",
        "airport",
        "station_id",
        "temp",
        "dwpt",
        "rhum",
        "prcp",
        "wdir",
        "wspd",
        "coco",
    ]

    weather = weather[columns_to_keep].copy()

    # -----------------------------------------------------
    # CLEAN TIME
    # -----------------------------------------------------

    weather["time"] = pd.to_datetime(
        weather["time"],
        errors="coerce",
    )

    # -----------------------------------------------------
    # CLEAN AIRPORT CODE
    # -----------------------------------------------------

    weather["airport"] = (
        weather["airport"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # -----------------------------------------------------
    # CONVERT WEATHER VARIABLES TO NUMERIC
    # -----------------------------------------------------

    weather_columns = [
        "temp",
        "dwpt",
        "rhum",
        "prcp",
        "wdir",
        "wspd",
        "coco",
    ]

    for column in weather_columns:
        weather[column] = pd.to_numeric(
            weather[column],
            errors="coerce",
        )

    # -----------------------------------------------------
    # REMOVE INVALID RECORDS
    # -----------------------------------------------------

    weather = weather.dropna(
        subset=[
            "time",
            "airport",
            "temp",
            "rhum",
            "wdir",
        ]
    )

    # -----------------------------------------------------
    # PRECIPITATION
    # -----------------------------------------------------
    # Keep missing precipitation as missing for now.
    # We do not automatically assume missing = zero.

    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    duplicate_count = weather.duplicated(
        subset=["airport", "time"]
    ).sum()

    print(
        f"Duplicate airport-hour records: "
        f"{duplicate_count:,}"
    )

    weather = weather.drop_duplicates(
        subset=["airport", "time"]
    )

    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    weather = weather.sort_values(
        ["airport", "time"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print(f"\nClean weather rows: {len(weather):,}")

    print(
        f"Airports represented: "
        f"{weather['airport'].nunique()}"
    )

    print("\nWeather records per airport:")

    print(
        weather["airport"]
        .value_counts()
        .sort_index()
    )

    print("\nMissing values after cleaning:")

    print(
        weather.isna().sum()
    )

    return weather


if __name__ == "__main__":

    cleaned_weather = clean_weather_data()

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        PROCESSED_DATA_DIR
        / "weather_clean.csv"
    )

    cleaned_weather.to_csv(
        output_file,
        index=False,
    )

    print("\n====================================")
    print("WEATHER CLEANING COMPLETE")
    print("====================================")

    print("\nCleaned weather data saved to:")
    print(output_file)