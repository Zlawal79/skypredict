import joblib
import pandas as pd

from src.config import MODEL_DIR


MODEL_FILE = MODEL_DIR / "random_forest.joblib"


def load_model():
    """
    Load the trained SkyPredict Random Forest model.
    """

    return joblib.load(MODEL_FILE)


def get_time_period(hour):
    """
    Convert scheduled departure hour into
    the time-period categories used during training.
    """

    if 5 <= hour < 12:
        return "morning"

    elif 12 <= hour < 17:
        return "afternoon"

    elif 17 <= hour < 22:
        return "evening"

    else:
        return "night"


def predict_delay(
    airline,
    origin,
    destination,
    day_of_week,
    day_of_month,
    departure_hour,
    scheduled_duration,
    distance,
    temperature,
    dew_point,
    humidity,
    precipitation,
    wind_direction,
    wind_speed,
    condition_code,
):
    """
    Predict the probability that a flight will experience
    a departure delay of 15 minutes or more.

    Returns
    -------
    dict
        Prediction class and delay probability.
    """

    model = load_model()

    # -----------------------------------------------------
    # DERIVED FEATURES
    # -----------------------------------------------------

    time_period = get_time_period(
        departure_hour
    )

    is_weekend = int(
        day_of_week in [6, 7]
    )

    # If precipitation is unavailable, preserve it as
    # missing so the trained preprocessing pipeline can
    # apply its imputation strategy.
    if precipitation is None:
        precipitation_value = float("nan")
        prcp_reported = 0

    else:
        precipitation_value = precipitation
        prcp_reported = 1

    # -----------------------------------------------------
    # BUILD MODEL INPUT
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "Reporting_Airline": airline,
                "Origin": origin,
                "Dest": destination,
                "DayOfWeek": day_of_week,
                "day_of_month": day_of_month,
                "departure_hour": departure_hour,
                "time_period": time_period,
                "is_weekend": is_weekend,
                "CRSElapsedTime": scheduled_duration,
                "Distance": distance,
                "temp": temperature,
                "dwpt": dew_point,
                "rhum": humidity,
                "prcp": precipitation_value,
                "prcp_reported": prcp_reported,
                "wdir": wind_direction,
                "wspd": wind_speed,
                "coco": condition_code,
            }
        ]
    )

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    delay_probability = (
        model.predict_proba(input_data)[0][1]
    )

    prediction = int(
        delay_probability >= 0.50
    )

    # -----------------------------------------------------
    # HUMAN-READABLE RESULT
    # -----------------------------------------------------

    if prediction == 1:
        label = "Delay Likely"
    else:
        label = "Delay Less Likely"

    return {
        "prediction": prediction,
        "label": label,
        "delay_probability": float(
            delay_probability
        ),
        "delay_probability_percent": float(
            delay_probability * 100
        ),
    }


if __name__ == "__main__":

    print("\nTesting SkyPredict prediction engine...")

    example = predict_delay(
        airline="AA",
        origin="JFK",
        destination="LAX",
        day_of_week=3,
        day_of_month=15,
        departure_hour=18,
        scheduled_duration=360,
        distance=2475,
        temperature=30,
        dew_point=20,
        humidity=55,
        precipitation=None,
        wind_direction=220,
        wind_speed=15,
        condition_code=3,
    )

    print("\nPrediction result:")

    print(example)