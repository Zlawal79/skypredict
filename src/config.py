from pathlib import Path

# Main SkyPredict project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model and report folders
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

# Dataset locations
FLIGHTS_FILE = RAW_DATA_DIR / "flights.csv"
WEATHER_FILE = RAW_DATA_DIR / "weather.csv"

# Processed dataset
MERGED_FILE = PROCESSED_DATA_DIR / "flight_weather.csv"

# Trained ML model
MODEL_FILE = MODEL_DIR / "flight_delay_model.joblib"

# Machine-learning settings
RANDOM_STATE = 42

# A flight is classified as delayed when the delay
# is 15 minutes or greater.
DELAY_THRESHOLD = 15