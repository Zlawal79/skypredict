import pandas as pd

from src.config import FLIGHTS_FILE, WEATHER_FILE


def load_flight_data(file_path=FLIGHTS_FILE):
    """
    Load the raw flight dataset.

    Parameters
    ----------
    file_path : Path
        Location of the flight CSV file.

    Returns
    -------
    pandas.DataFrame
        Raw flight data.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nFlight dataset was not found at:\n{file_path}\n"
            "Place the flight CSV inside data/raw/ and name it flights.csv."
        )

    flights = pd.read_csv(file_path)

    print("\n--- FLIGHT DATA LOADED ---")
    print(f"Rows: {flights.shape[0]:,}")
    print(f"Columns: {flights.shape[1]}")
    print("\nFlight columns:")
    print(flights.columns.tolist())

    return flights


def load_weather_data(file_path=WEATHER_FILE):
    """
    Load the raw weather dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nWeather dataset was not found at:\n{file_path}\n"
            "Place the weather CSV inside data/raw/ and name it weather.csv."
        )

    weather = pd.read_csv(file_path)

    print("\n--- WEATHER DATA LOADED ---")
    print(f"Rows: {weather.shape[0]:,}")
    print(f"Columns: {weather.shape[1]}")
    print("\nWeather columns:")
    print(weather.columns.tolist())

    return weather


if __name__ == "__main__":
    print("Testing SkyPredict data loader...")

    try:
        load_flight_data()
    except FileNotFoundError as error:
        print(error)

    try:
        load_weather_data()
    except FileNotFoundError as error:
        print(error)