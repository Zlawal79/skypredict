from datetime import datetime
from pathlib import Path

import pandas as pd
from meteostat import Hourly


# =========================================================
# SKYPREDICT WEATHER DOWNLOADER
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "weather.csv"


# July 2026 - matches the BTS flight dataset
START_DATE = datetime(2026, 7, 1, 0, 0)
END_DATE = datetime(2026, 7, 31, 23, 59)


# ---------------------------------------------------------
# AIRPORT -> WEATHER STATION
# ---------------------------------------------------------
# We are starting with major U.S. airports so that we can
# build and validate the full SkyPredict ML pipeline.
# ---------------------------------------------------------

AIRPORT_STATIONS = {
    "ATL": "72219",
    "JFK": "74486",
    "LGA": "72503",
    "ORD": "72530",
    "DFW": "72259",
    "DEN": "72565",
    "LAX": "72295",
    "SFO": "72494",
    "SEA": "72793",
    "MIA": "72202",
    "BOS": "72509",
    "PHX": "72278",
    "LAS": "72386",
    "MSP": "72658",
    "DTW": "72537",
    "PHL": "72408",
    "CLT": "72314",
    "IAH": "72243",
    "MCO": "72205",
    "BWI": "72406",
}


def download_airport_weather():

    print("\n====================================")
    print("SKYPREDICT WEATHER DOWNLOAD")
    print("====================================")

    print(f"\nStart date: {START_DATE}")
    print(f"End date:   {END_DATE}")
    print(f"Airports to download: {len(AIRPORT_STATIONS)}")

    all_weather = []

    successful_airports = []
    failed_airports = []

    # -----------------------------------------------------
    # DOWNLOAD WEATHER
    # -----------------------------------------------------

    for airport, station_id in AIRPORT_STATIONS.items():

        print(
            f"\nDownloading weather for "
            f"{airport} (station {station_id})..."
        )

        try:

            # Meteostat 1.7.6 Hourly interface
            weather = Hourly(
                station_id,
                START_DATE,
                END_DATE,
            ).fetch()

            if weather.empty:

                print(
                    f"WARNING: No weather data returned "
                    f"for {airport}"
                )

                failed_airports.append(airport)
                continue

            # Time is originally the DataFrame index
            weather = weather.reset_index()

            # Store the airport and station used
            weather["airport"] = airport
            weather["station_id"] = station_id

            all_weather.append(weather)

            successful_airports.append(airport)

            print(
                f"SUCCESS: {len(weather):,} "
                f"hourly weather records"
            )

        except Exception as error:

            print(
                f"ERROR downloading {airport}: "
                f"{error}"
            )

            failed_airports.append(airport)

    # -----------------------------------------------------
    # MAKE SURE DATA EXISTS
    # -----------------------------------------------------

    if not all_weather:

        print("\n====================================")
        print("NO WEATHER DATA DOWNLOADED")
        print("====================================")

        return

    # -----------------------------------------------------
    # COMBINE WEATHER
    # -----------------------------------------------------

    combined_weather = pd.concat(
        all_weather,
        ignore_index=True,
    )

    combined_weather = combined_weather.sort_values(
        by=["airport", "time"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # SAVE WEATHER CSV
    # -----------------------------------------------------

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_weather.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n====================================")
    print("WEATHER DOWNLOAD COMPLETE")
    print("====================================")

    print(
        f"\nTotal weather records: "
        f"{len(combined_weather):,}"
    )

    print(
        f"Successful airports: "
        f"{len(successful_airports)}"
    )

    print(
        f"Failed airports: "
        f"{len(failed_airports)}"
    )

    if successful_airports:

        print("\nSuccessful airports:")

        print(
            ", ".join(successful_airports)
        )

    if failed_airports:

        print("\nFailed airports:")

        print(
            ", ".join(failed_airports)
        )

    print("\nWeather columns:")

    print(
        combined_weather.columns.tolist()
    )

    print("\nFirst five weather records:")

    print(
        combined_weather.head()
    )

    print("\nMissing values:")

    print(
        combined_weather.isna().sum()
    )

    print("\nWeather data saved to:")

    print(OUTPUT_FILE)


if __name__ == "__main__":

    download_airport_weather()