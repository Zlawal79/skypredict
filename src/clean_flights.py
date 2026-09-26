import pandas as pd

from src.config import FLIGHTS_FILE, PROCESSED_DATA_DIR


# Columns needed from the BTS flight dataset
FLIGHT_COLUMNS = [
    "Year",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "OriginCityName",
    "OriginState",
    "Dest",
    "DestCityName",
    "DestState",
    "CRSDepTime",
    "CRSArrTime",
    "DepDelayMinutes",
    "DepDel15",
    "DepTimeBlk",
    "Cancelled",
    "Diverted",
    "CRSElapsedTime",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]


def clean_flight_data(file_path=FLIGHTS_FILE):
    """
    Load and clean the BTS flight dataset for SkyPredict.

    The cleaned dataset will later be used for:
    - Exploratory data analysis
    - Weather-data integration
    - Feature engineering
    - Flight-delay prediction

    DepDel15 is the prediction target:
        0 = departure delay under 15 minutes
        1 = departure delay of 15 minutes or more
    """

    print("\nLoading selected flight columns...")

    # Load only the columns SkyPredict needs instead of all 110 columns
    flights = pd.read_csv(
        file_path,
        usecols=FLIGHT_COLUMNS,
        low_memory=False,
    )

    print(f"Raw rows: {len(flights):,}")
    print(f"Selected columns: {len(flights.columns)}")

    # ---------------------------------------------------------
    # 1. CLEAN FLIGHT DATE
    # ---------------------------------------------------------

    flights["FlightDate"] = pd.to_datetime(
        flights["FlightDate"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # 2. REMOVE CANCELLED AND DIVERTED FLIGHTS
    # ---------------------------------------------------------
    # SkyPredict currently predicts departure delays for flights
    # that actually operated.

    flights = flights[
        (flights["Cancelled"] == 0)
        & (flights["Diverted"] == 0)
    ].copy()

    # ---------------------------------------------------------
    # 3. REMOVE ROWS WITHOUT A VALID TARGET
    # ---------------------------------------------------------

    flights = flights.dropna(
        subset=[
            "FlightDate",
            "DepDel15",
            "DepDelayMinutes",
        ]
    )

    # ---------------------------------------------------------
    # 4. CLEAN TARGET VARIABLE
    # ---------------------------------------------------------

    flights["DepDel15"] = flights["DepDel15"].astype(int)

    # ---------------------------------------------------------
    # 5. CLEAN AIRLINE AND AIRPORT CODES
    # ---------------------------------------------------------

    code_columns = [
        "Reporting_Airline",
        "Origin",
        "Dest",
    ]

    for column in code_columns:
        flights[column] = (
            flights[column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # ---------------------------------------------------------
    # 6. REMOVE DUPLICATES
    # ---------------------------------------------------------

    duplicate_count = flights.duplicated().sum()

    print(f"Duplicate rows found: {duplicate_count:,}")

    flights = flights.drop_duplicates()

    # ---------------------------------------------------------
    # 7. SORT FLIGHTS CHRONOLOGICALLY
    # ---------------------------------------------------------

    flights = flights.sort_values(
        by=[
            "FlightDate",
            "CRSDepTime",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 8. DISPLAY CLEANING SUMMARY
    # ---------------------------------------------------------

    print(f"\nClean rows: {len(flights):,}")

    print("\nDelay distribution:")
    print(
        flights["DepDel15"]
        .value_counts()
        .sort_index()
    )

    print("\nDelay percentages:")
    print(
        flights["DepDel15"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nMissing values in important columns:")

    important_columns = [
        "FlightDate",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "CRSDepTime",
        "DepDelayMinutes",
        "DepDel15",
        "Distance",
    ]

    print(
        flights[important_columns]
        .isna()
        .sum()
    )

    return flights


if __name__ == "__main__":

    cleaned_flights = clean_flight_data()

    # Location for the cleaned flight dataset
    output_file = PROCESSED_DATA_DIR / "flights_clean.csv"

    # Make sure the processed-data folder exists
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save cleaned dataset
    cleaned_flights.to_csv(
        output_file,
        index=False,
    )

    print("\n--------------------------------")
    print("FLIGHT CLEANING COMPLETE")
    print("--------------------------------")

    print("\nCleaned flight data saved to:")
    print(output_file)