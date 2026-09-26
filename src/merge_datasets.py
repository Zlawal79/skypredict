import pandas as pd

from src.config import PROCESSED_DATA_DIR, MERGED_FILE


FLIGHTS_FILE = PROCESSED_DATA_DIR / "flights_clean.csv"
WEATHER_FILE = PROCESSED_DATA_DIR / "weather_clean.csv"


def convert_crs_time(value):
    """
    Convert BTS CRSDepTime (HHMM) into hour and minute.

    Examples:
        530  -> 05:30
        1430 -> 14:30
        2400 -> 00:00
    """

    if pd.isna(value):
        return None

    try:
        value = int(value)
    except (ValueError, TypeError):
        return None

    # BTS can occasionally represent midnight as 2400
    if value == 2400:
        return 0, 0

    hour = value // 100
    minute = value % 100

    if hour > 23 or minute > 59:
        return None

    return hour, minute


def build_scheduled_departure(row):
    """
    Combine FlightDate and CRSDepTime into one timestamp.
    """

    flight_date = row["FlightDate"]

    time_parts = convert_crs_time(
        row["CRSDepTime"]
    )

    if pd.isna(flight_date) or time_parts is None:
        return pd.NaT

    hour, minute = time_parts

    return (
        flight_date
        + pd.Timedelta(hours=hour)
        + pd.Timedelta(minutes=minute)
    )


def merge_flight_weather():

    print("\n====================================")
    print("SKYPREDICT FLIGHT + WEATHER MERGE")
    print("====================================")

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    print("\nLoading cleaned flight data...")

    flights = pd.read_csv(
        FLIGHTS_FILE,
        parse_dates=["FlightDate"],
    )

    print(
        f"Flight rows loaded: "
        f"{len(flights):,}"
    )

    print("\nLoading cleaned weather data...")

    weather = pd.read_csv(
        WEATHER_FILE,
        parse_dates=["time"],
    )

    print(
        f"Weather rows loaded: "
        f"{len(weather):,}"
    )

    # -----------------------------------------------------
    # KEEP FLIGHTS FROM WEATHER-COVERED AIRPORTS
    # -----------------------------------------------------

    weather_airports = set(
        weather["airport"].unique()
    )

    flights = flights[
        flights["Origin"].isin(
            weather_airports
        )
    ].copy()

    print(
        f"\nFlights from weather-covered airports: "
        f"{len(flights):,}"
    )

    print(
        f"Covered origin airports: "
        f"{flights['Origin'].nunique()}"
    )

    # -----------------------------------------------------
    # BUILD SCHEDULED DEPARTURE TIMESTAMP
    # -----------------------------------------------------

    print(
        "\nBuilding scheduled departure timestamps..."
    )

    flights["scheduled_departure"] = flights.apply(
        build_scheduled_departure,
        axis=1,
    )

    invalid_times = (
        flights["scheduled_departure"]
        .isna()
        .sum()
    )

    print(
        f"Invalid scheduled departure timestamps: "
        f"{invalid_times:,}"
    )

    flights = flights.dropna(
        subset=["scheduled_departure"]
    )

    # -----------------------------------------------------
    # ROUND FLIGHT TIME TO NEAREST HOUR
    # -----------------------------------------------------
    # Weather data is hourly.

    flights["weather_hour"] = (
        flights["scheduled_departure"]
        .dt.round("h")
    )

    # -----------------------------------------------------
    # PREPARE WEATHER
    # -----------------------------------------------------

    weather = weather.rename(
        columns={
            "airport": "Origin",
            "time": "weather_hour",
        }
    )

    # -----------------------------------------------------
    # MERGE
    # -----------------------------------------------------

    print(
        "\nMatching flights with hourly weather..."
    )

    merged = flights.merge(
        weather,
        on=[
            "Origin",
            "weather_hour",
        ],
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------
    # CHECK WEATHER MATCH RATE
    # -----------------------------------------------------

    matched = merged["temp"].notna().sum()

    unmatched = merged["temp"].isna().sum()

    match_rate = (
        matched / len(merged) * 100
        if len(merged) > 0
        else 0
    )

    print(
        f"\nFlights with weather match: "
        f"{matched:,}"
    )

    print(
        f"Flights without weather match: "
        f"{unmatched:,}"
    )

    print(
        f"Weather match rate: "
        f"{match_rate:.2f}%"
    )

    # -----------------------------------------------------
    # KEEP ONLY MATCHED FLIGHTS
    # -----------------------------------------------------

    merged = merged[
        merged["temp"].notna()
    ].copy()

    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    merged = merged.sort_values(
        [
            "scheduled_departure",
            "Origin",
        ]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        MERGED_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n====================================")
    print("MERGE COMPLETE")
    print("====================================")

    print(
        f"\nFinal merged rows: "
        f"{len(merged):,}"
    )

    print(
        f"Final columns: "
        f"{len(merged.columns)}"
    )

    print(
        f"Origin airports: "
        f"{merged['Origin'].nunique()}"
    )

    print("\nDelay distribution:")

    print(
        merged["DepDel15"]
        .value_counts()
        .sort_index()
    )

    print("\nDelay percentages:")

    print(
        merged["DepDel15"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nMerged dataset saved to:")

    print(MERGED_FILE)


if __name__ == "__main__":

    merge_flight_weather()