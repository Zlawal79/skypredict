from pathlib import Path
import sys
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_delay


MODEL_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_data.csv"
RESULTS_FILE = PROJECT_ROOT / "models" / "model_results.json"
IMPORTANCE_FILE = (
    PROJECT_ROOT / "reports" / "figures" / "feature_importance.csv"
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SkyPredict",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #F6F8F3;
        color: #17251E;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 5rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #123D2C !important;
        letter-spacing: -0.03em;
    }

    p, label {
        color: #435249;
    }


    /* SIDEBAR */

    section[data-testid="stSidebar"] {
        background: #103F2E;
    }

    section[data-testid="stSidebar"] * {
        color: #F4F7F2 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }


    /* BUTTONS */

    div.stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 10px;
        border: none;
        background: #176A47;
        color: white;
        font-weight: 700;
        font-size: 0.98rem;
    }

    div.stButton > button:hover {
        background: #105438;
        color: white;
        border: none;
    }


    /* INPUTS */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: white !important;
        color: #17251E !important;
    }

    div[data-baseweb="select"] span {
        color: #17251E !important;
    }

    input {
        color: #17251E !important;
    }

    .stNumberInput label,
    .stSelectbox label,
    .stSlider label,
    .stCheckbox label {
        color: #25372D !important;
        font-weight: 600 !important;
    }


    /* METRICS */

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #DCE6DF;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0px 4px 15px rgba(25,60,40,0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #607068 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #14583D !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }


    /* EXPANDERS */

    details {
        background: white !important;
        border: 1px solid #DCE6DF !important;
        border-radius: 12px !important;
    }


    /* TABS */

    button[data-baseweb="tab"] {
        font-weight: 700;
    }


    /* PROGRESS */

    div[data-testid="stProgress"] > div > div > div {
        background-color: #1E7B53;
    }


    /* REMOVE DEFAULT STREAMLIT ITEMS */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD PROJECT DATA
# =========================================================

@st.cache_data
def load_model_data():
    return pd.read_csv(MODEL_DATA_FILE)


@st.cache_data
def load_results():
    with open(RESULTS_FILE, "r") as file:
        return json.load(file)


@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_FILE)


model_data = load_model_data()
results = load_results()
importance = load_importance()

forest = results["random_forest"]
logistic = results["logistic_regression"]


# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "About & How It Works"


def go_to(page_name):
    st.session_state.page = page_name


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def section_intro(kicker, title, description=None):
    st.caption(kicker.upper())
    st.header(title)

    if description:
        st.write(description)


def clean_feature_name(name):
    name = name.replace("numeric__", "")
    name = name.replace("categorical__", "")

    mapping = {
        "departure_hour": "Departure time",
        "time_period_morning": "Morning flights",
        "time_period_evening": "Evening flights",
        "time_period_afternoon": "Afternoon flights",
        "time_period_night": "Night flights",
        "temp": "Temperature",
        "coco": "Weather conditions",
        "rhum": "Humidity",
        "dwpt": "Dew point",
        "day_of_month": "Day of month",
        "wspd": "Wind speed",
        "wdir": "Wind direction",
        "CRSElapsedTime": "Scheduled duration",
        "Distance": "Flight distance",
        "DayOfWeek": "Day of week",
        "Reporting_Airline_WN": "Southwest Airlines",
        "Reporting_Airline_AA": "American Airlines",
        "Reporting_Airline_MQ": "Envoy Air",
        "Reporting_Airline_OO": "SkyWest Airlines",
        "Reporting_Airline_UA": "United Airlines",
        "Dest_SFO": "Destination: San Francisco",
    }

    return mapping.get(
        name,
        name.replace("_", " ").title()
    )


def time_period_from_hour(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    return "Night"


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("SkyPredict")
st.sidebar.write("Flight Delay Prediction")

st.sidebar.divider()

st.sidebar.caption("EXPLORE")

pages = [
    "About & How It Works",
    "Try It Out",
    "Results & Insights",
]

for page_name in pages:
    if st.sidebar.button(
        page_name,
        key=f"nav_{page_name}",
        use_container_width=True,
    ):
        go_to(page_name)

st.sidebar.divider()

st.sidebar.caption(
    "Built from real flight-performance records "
    "and historical hourly weather."
)

st.sidebar.caption(
    "Target: departure delay of 15 minutes or more."
)


# =========================================================
# PAGE 1
# ABOUT + HOW IT WORKS
# =========================================================

if st.session_state.page == "About & How It Works":

    st.caption("SKYPREDICT")

    st.title(
        "Can we predict a flight delay before departure?"
    )

    st.markdown(
        """
        ### Turning flight schedules and weather conditions
        into an understandable estimate of delay risk.
        """
    )

    st.write(
        """
        SkyPredict is an end-to-end data science project that
        combines real flight-performance records with historical
        weather observations to explore whether the conditions
        surrounding a flight can help identify departure-delay risk.
        """
    )

    hero1, hero2 = st.columns([1, 1])

    with hero1:
        if st.button(
            "Try SkyPredict",
            key="hero_try",
        ):
            go_to("Try It Out")
            st.rerun()

    with hero2:
        if st.button(
            "Explore the Results",
            key="hero_results",
        ):
            go_to("Results & Insights")
            st.rerun()

    st.divider()

    # -----------------------------------------------------
    # PROJECT SCALE
    # -----------------------------------------------------

    section_intro(
        "The Project at a Glance",
        "Built from hundreds of thousands of real flights",
        (
            "The project begins with raw aviation data and "
            "turns it into a weather-linked prediction dataset."
        ),
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Raw Flight Records",
        "631,970"
    )

    m2.metric(
        "Flight + Weather Records",
        f"{results['dataset_rows']:,}"
    )

    m3.metric(
        "Origin Airports",
        "20"
    )

    m4.metric(
        "Unseen Test Flights",
        f"{results['testing_rows']:,}"
    )

    st.divider()

    # -----------------------------------------------------
    # MOTIVATION
    # -----------------------------------------------------

    section_intro(
        "Motivation",
        "Why I built SkyPredict",
    )

    with st.container(border=True):

        st.subheader(
            "Flight delays look simple. "
            "The conditions behind them are not."
        )

        st.write(
            """
            A delayed departure can be associated with many
            different pieces of information: when the flight
            leaves, where it is going, the airline operating it,
            the length of the route and the weather around the
            departure airport.
            """
        )

        st.write(
            """
            I built SkyPredict to investigate one question:
            **can these signals be combined to identify flights
            with higher delay risk before departure?**
            """
        )

        st.write(
            """
            Rather than stopping at a machine-learning model,
            I wanted to build the entire workflow — preparing
            large public datasets, connecting flight records with
            weather, testing different prediction approaches,
            evaluating them on unseen flights and turning the
            result into an application that anyone can explore.
            """
        )

    st.divider()

    # -----------------------------------------------------
    # WHAT THE PROJECT DOES
    # -----------------------------------------------------

    section_intro(
        "The Idea",
        "What does SkyPredict actually do?",
        (
            "At its core, SkyPredict compares the conditions "
            "around a new flight with patterns found in previous flights."
        ),
    )

    idea1, idea2 = st.columns(2)

    with idea1:

        with st.container(border=True):

            st.subheader("Information going in")

            st.write(
                """
                SkyPredict considers information such as:
                """
            )

            st.markdown(
                """
                - airline
                - origin and destination
                - scheduled departure time
                - flight distance and duration
                - temperature
                - humidity
                - wind conditions
                - general weather conditions
                """
            )

    with idea2:

        with st.container(border=True):

            st.subheader("Prediction coming out")

            st.write(
                """
                The application returns an estimated probability
                that the flight will depart **15 minutes or more
                behind schedule**.
                """
            )

            st.write(
                """
                The result is a probability, not a guarantee.
                It represents patterns found in the historical
                data used to build the project.
                """
            )

    st.divider()

    # -----------------------------------------------------
    # HOW IT WORKS
    # -----------------------------------------------------

    section_intro(
        "From Data to Prediction",
        "How SkyPredict works",
        (
            "No code is needed to understand the workflow. "
            "The project moves through six main stages."
        ),
    )

    row1 = st.columns(3)

    with row1[0]:
        with st.container(border=True):
            st.caption("01")
            st.subheader("Collect")
            st.write(
                "Start with 631,970 real flight-performance records."
            )

    with row1[1]:
        with st.container(border=True):
            st.caption("02")
            st.subheader("Prepare")
            st.write(
                "Clean the data and remove cancelled, diverted "
                "and unusable flight records."
            )

    with row1[2]:
        with st.container(border=True):
            st.caption("03")
            st.subheader("Connect")
            st.write(
                "Match flights with hourly weather near the "
                "origin airport and scheduled departure time."
            )

    row2 = st.columns(3)

    with row2[0]:
        with st.container(border=True):
            st.caption("04")
            st.subheader("Learn")
            st.write(
                "Train prediction models to find patterns "
                "associated with delayed flights."
            )

    with row2[1]:
        with st.container(border=True):
            st.caption("05")
            st.subheader("Test")
            st.write(
                f"Evaluate the models using {results['testing_rows']:,} "
                "flights they did not see during training."
            )

    with row2[2]:
        with st.container(border=True):
            st.caption("06")
            st.subheader("Predict")
            st.write(
                "Turn the final model into an interactive tool "
                "for exploring new flight scenarios."
            )

    st.divider()

    # -----------------------------------------------------
    # TECHNOLOGY
    # -----------------------------------------------------

    section_intro(
        "Under the Hood",
        "How it was built",
        (
            "The interface is simple, but the project behind it "
            "covers data engineering, machine learning and deployment."
        ),
    )

    t1, t2, t3 = st.columns(3)

    with t1:
        with st.container(border=True):
            st.subheader("Data")
            st.write(
                "Python, Pandas, public flight-performance data "
                "and Meteostat historical weather."
            )

    with t2:
        with st.container(border=True):
            st.subheader("Prediction")
            st.write(
                "Logistic Regression and Random Forest models "
                "were trained and compared."
            )

    with t3:
        with st.container(border=True):
            st.subheader("Application")
            st.write(
                "Streamlit and Plotly turn the final workflow "
                "into an interactive experience."
            )

    st.divider()

    # -----------------------------------------------------
    # CTA
    # -----------------------------------------------------

    st.header("See SkyPredict in action")

    st.write(
        """
        Build a flight scenario, enter the weather conditions
        and see the delay probability estimated by the model.
        """
    )

    if st.button(
        "Try the Flight Predictor",
        key="bottom_try",
    ):
        go_to("Try It Out")
        st.rerun()


# =========================================================
# PAGE 2
# TRY IT OUT
# =========================================================

elif st.session_state.page == "Try It Out":

    st.caption("INTERACTIVE DEMO")

    st.title("Try SkyPredict")

    st.write(
        """
        Build a flight scenario and see the estimated probability
        that it will depart at least 15 minutes behind schedule.
        """
    )

    with st.container(border=True):

        st.subheader("How to use it")

        i1, i2, i3 = st.columns(3)

        with i1:
            st.markdown("**1. Build the flight**")
            st.write(
                "Choose the airline, route and schedule."
            )

        with i2:
            st.markdown("**2. Add the conditions**")
            st.write(
                "Enter the weather expected at departure."
            )

        with i3:
            st.markdown("**3. Run the prediction**")
            st.write(
                "SkyPredict estimates the delay probability."
            )

    st.divider()

    # -----------------------------------------------------
    # OPTIONS
    # -----------------------------------------------------

    airline_options = sorted(
        model_data["Reporting_Airline"]
        .dropna()
        .unique()
    )

    origin_options = sorted(
        model_data["Origin"]
        .dropna()
        .unique()
    )

    destination_options = sorted(
        model_data["Dest"]
        .dropna()
        .unique()
    )

    # -----------------------------------------------------
    # FLIGHT
    # -----------------------------------------------------

    section_intro(
        "Step 1",
        "Build the flight",
        "Enter the planned flight information.",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        airline = st.selectbox(
            "Airline",
            airline_options,
        )

    with c2:
        origin = st.selectbox(
            "Origin Airport",
            origin_options,
        )

    with c3:
        destination = st.selectbox(
            "Destination Airport",
            destination_options,
        )

    c4, c5, c6 = st.columns(3)

    with c4:
        day_of_week = st.selectbox(
            "Day of Week",
            [1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: {
                1: "Monday",
                2: "Tuesday",
                3: "Wednesday",
                4: "Thursday",
                5: "Friday",
                6: "Saturday",
                7: "Sunday",
            }[x],
        )

    with c5:
        day_of_month = st.number_input(
            "Day of Month",
            min_value=1,
            max_value=31,
            value=15,
        )

    with c6:
        departure_hour = st.slider(
            "Scheduled Departure Hour",
            0,
            23,
            12,
            format="%d:00",
        )

    c7, c8 = st.columns(2)

    with c7:
        scheduled_duration = st.number_input(
            "Scheduled Flight Duration (minutes)",
            min_value=20,
            max_value=1000,
            value=180,
            step=5,
        )

    with c8:
        distance = st.number_input(
            "Flight Distance (miles)",
            min_value=50,
            max_value=6000,
            value=1000,
            step=10,
        )

    st.divider()

    # -----------------------------------------------------
    # WEATHER
    # -----------------------------------------------------

    section_intro(
        "Step 2",
        "Add departure weather",
        (
            "Enter the conditions around the origin airport. "
            "These values become part of the prediction."
        ),
    )

    w1, w2, w3 = st.columns(3)

    with w1:

        temperature = st.number_input(
            "Temperature (°C)",
            -50.0,
            60.0,
            25.0,
            step=0.5,
        )

        dew_point = st.number_input(
            "Dew Point (°C)",
            -60.0,
            50.0,
            15.0,
            step=0.5,
        )

    with w2:

        humidity = st.slider(
            "Relative Humidity (%)",
            0,
            100,
            60,
        )

        wind_speed = st.number_input(
            "Wind Speed (km/h)",
            0.0,
            200.0,
            10.0,
            step=1.0,
        )

    with w3:

        wind_direction = st.number_input(
            "Wind Direction (degrees)",
            0,
            360,
            180,
        )

        weather_name = st.selectbox(
            "General Weather",
            [
                "Clear",
                "Mostly Clear",
                "Cloudy",
                "Overcast",
                "Fog",
                "Rain",
                "Heavy Rain",
                "Thunderstorm",
            ],
        )

    weather_codes = {
        "Clear": 1,
        "Mostly Clear": 2,
        "Cloudy": 3,
        "Overcast": 4,
        "Fog": 5,
        "Rain": 7,
        "Heavy Rain": 9,
        "Thunderstorm": 25,
    }

    condition_code = weather_codes[weather_name]

    precipitation_available = st.checkbox(
        "Add a precipitation measurement"
    )

    precipitation = None

    if precipitation_available:

        precipitation = st.number_input(
            "Precipitation (mm)",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=0.1,
        )

    st.divider()

    # -----------------------------------------------------
    # RUN
    # -----------------------------------------------------

    section_intro(
        "Step 3",
        "Run the prediction",
        (
            "SkyPredict will compare this scenario with the "
            "patterns learned from historical flights."
        ),
    )

    if st.button(
        "Run SkyPredict",
        key="predict_button",
    ):

        result = predict_delay(
            airline=airline,
            origin=origin,
            destination=destination,
            day_of_week=day_of_week,
            day_of_month=day_of_month,
            departure_hour=departure_hour,
            scheduled_duration=scheduled_duration,
            distance=distance,
            temperature=temperature,
            dew_point=dew_point,
            humidity=humidity,
            precipitation=precipitation,
            wind_direction=wind_direction,
            wind_speed=wind_speed,
            condition_code=condition_code,
        )

        probability = result["delay_probability_percent"]

        if probability < 30:
            risk = "Lower"
        elif probability < 50:
            risk = "Moderate"
        elif probability < 70:
            risk = "Elevated"
        else:
            risk = "Higher"

        st.divider()

        st.caption("YOUR PREDICTION")

        st.header(
            f"{probability:.1f}% estimated chance of delay"
        )

        st.subheader(
            f"{risk} delay risk"
        )

        st.progress(
            min(probability / 100, 1.0)
        )

        st.write(
            """
            This is the estimated probability that the flight
            will depart **15 minutes or more behind schedule**.
            """
        )

        # -------------------------------------------------
        # GAUGE
        # -------------------------------------------------

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                number={
                    "suffix": "%",
                    "font": {
                        "size": 48,
                        "color": "#123D2C",
                    },
                },
                title={
                    "text": "Estimated Delay Risk",
                    "font": {
                        "size": 18,
                        "color": "#435249",
                    },
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                    },
                    "bar": {
                        "color": "#176A47",
                    },
                    "bgcolor": "#E8EFEA",
                    "borderwidth": 0,
                    "steps": [
                        {
                            "range": [0, 30],
                            "color": "#DCEDE2",
                        },
                        {
                            "range": [30, 50],
                            "color": "#C5DEC9",
                        },
                        {
                            "range": [50, 70],
                            "color": "#A8CDB4",
                        },
                        {
                            "range": [70, 100],
                            "color": "#82B596",
                        },
                    ],
                },
            )
        )

        gauge.update_layout(
            height=340,
            margin=dict(
                l=30,
                r=30,
                t=60,
                b=20,
            ),
            paper_bgcolor="#F6F8F3",
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )

        # -------------------------------------------------
        # SCENARIO SUMMARY
        # -------------------------------------------------

        st.subheader("Scenario analyzed")

        s1, s2, s3 = st.columns(3)

        with s1:
            with st.container(border=True):
                st.caption("FLIGHT")
                st.subheader(
                    f"{origin} → {destination}"
                )
                st.write(
                    f"Airline: {airline}"
                )
                st.write(
                    f"Distance: {distance:,} miles"
                )

        with s2:
            with st.container(border=True):
                st.caption("SCHEDULE")
                st.subheader(
                    f"{departure_hour:02d}:00"
                )
                st.write(
                    f"{time_period_from_hour(departure_hour)} departure"
                )
                st.write(
                    f"{scheduled_duration} minute scheduled duration"
                )

        with s3:
            with st.container(border=True):
                st.caption("WEATHER")
                st.subheader(weather_name)
                st.write(
                    f"{temperature:.1f}°C · {humidity}% humidity"
                )
                st.write(
                    f"{wind_speed:.1f} km/h wind"
                )

        # -------------------------------------------------
        # EXPLANATION
        # -------------------------------------------------

        with st.container(border=True):

            st.subheader(
                "How should I interpret this?"
            )

            st.write(
                f"""
                SkyPredict estimated a **{probability:.1f}%**
                probability for this scenario. That does not
                mean the flight will definitely be delayed.
                """
            )

            st.write(
                """
                The result means that this combination of
                schedule, route and weather information resembles
                patterns that the model learned from historical
                delayed and non-delayed flights.
                """
            )

        if st.button(
            "Explore the Results Behind SkyPredict",
            key="prediction_results",
        ):
            go_to("Results & Insights")
            st.rerun()


# =========================================================
# PAGE 3
# RESULTS + INSIGHTS
# =========================================================

elif st.session_state.page == "Results & Insights":

    st.caption("RESULTS & INSIGHTS")

    st.title("What did SkyPredict discover?")

    st.write(
        """
        Explore the flight data, model performance and the
        information that contributed most to SkyPredict's predictions.
        """
    )

    # -----------------------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------------------

    section_intro(
        "The Dataset",
        "What did the model learn from?",
    )

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Flight + Weather Records",
        f"{results['dataset_rows']:,}"
    )

    d2.metric(
        "Delayed Flights",
        "31.48%"
    )

    d3.metric(
        "Not Delayed",
        "68.52%"
    )

    st.divider()

    # -----------------------------------------------------
    # DELAY DISTRIBUTION
    # -----------------------------------------------------

    section_intro(
        "Finding 1",
        "Most flights were not delayed",
        (
            "A little under one-third of the weather-linked "
            "flights in the model dataset had a departure "
            "delay of at least 15 minutes."
        ),
    )

    delay_counts = (
        model_data["DepDel15"]
        .value_counts()
        .sort_index()
    )

    delay_df = pd.DataFrame(
        {
            "Flight Status": [
                "Not Delayed",
                "Delayed 15+ Minutes",
            ],
            "Flights": [
                delay_counts.get(0, 0),
                delay_counts.get(1, 0),
            ],
        }
    )

    fig_delay = px.bar(
        delay_df,
        x="Flight Status",
        y="Flights",
        text="Flights",
    )

    fig_delay.update_traces(
        marker_color=[
            "#A7CDB4",
            "#176A47",
        ],
        texttemplate="%{text:,}",
        textposition="outside",
    )

    fig_delay.update_layout(
        title="Delay distribution in the model dataset",
        height=430,
        showlegend=False,
        paper_bgcolor="#F6F8F3",
        plot_bgcolor="#F6F8F3",
        yaxis_title="Number of flights",
        xaxis_title="",
    )

    st.plotly_chart(
        fig_delay,
        use_container_width=True,
    )

    st.write(
        """
        **Why this matters:** because delayed flights are the
        smaller group, simply looking at overall accuracy can be
        misleading. SkyPredict therefore uses several evaluation
        measures.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # DEPARTURE HOUR DELAY RATE
    # -----------------------------------------------------

    section_intro(
        "Finding 2",
        "Does departure time matter?",
        (
            "Departure hour was the strongest individual feature "
            "in the trained Random Forest."
        ),
    )

    hourly = (
        model_data
        .groupby("departure_hour")["DepDel15"]
        .mean()
        .mul(100)
        .reset_index()
    )

    hourly.columns = [
        "Departure Hour",
        "Delay Rate",
    ]

    fig_hour = px.line(
        hourly,
        x="Departure Hour",
        y="Delay Rate",
        markers=True,
    )

    fig_hour.update_traces(
        line_color="#176A47",
        marker_color="#176A47",
        line_width=4,
    )

    fig_hour.update_layout(
        title="Observed delay rate by scheduled departure hour",
        xaxis_title="Scheduled departure hour",
        yaxis_title="Flights delayed 15+ minutes (%)",
        height=450,
        paper_bgcolor="#F6F8F3",
        plot_bgcolor="#F6F8F3",
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True,
    )

    st.write(
        """
        This chart shows the **observed delay rate in the
        project dataset**, rather than a model prediction.
        It helps explain why departure time provided useful
        information to SkyPredict.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # AIRPORT DELAY RATES
    # -----------------------------------------------------

    section_intro(
        "Finding 3",
        "How did delay rates vary across origin airports?",
        (
            "SkyPredict's weather-linked dataset contains flights "
            "originating from 20 major U.S. airports."
        ),
    )

    airport_rates = (
        model_data
        .groupby("Origin")["DepDel15"]
        .agg(["mean", "count"])
        .reset_index()
    )

    airport_rates["Delay Rate"] = (
        airport_rates["mean"] * 100
    )

    airport_rates = airport_rates.sort_values(
        "Delay Rate",
        ascending=True,
    )

    fig_airports = px.bar(
        airport_rates,
        x="Delay Rate",
        y="Origin",
        orientation="h",
        text="Delay Rate",
    )

    fig_airports.update_traces(
        marker_color="#2D805A",
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig_airports.update_layout(
        title="Observed delay rate by origin airport",
        xaxis_title="Flights delayed 15+ minutes (%)",
        yaxis_title="Origin airport",
        height=650,
        paper_bgcolor="#F6F8F3",
        plot_bgcolor="#F6F8F3",
    )

    st.plotly_chart(
        fig_airports,
        use_container_width=True,
    )

    st.write(
        """
        These percentages describe this specific July 2026
        dataset. They should not be interpreted as permanent
        rankings of airport performance.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    section_intro(
        "Finding 4",
        "What information influenced the model most?",
        (
            "Random Forest feature importance shows which pieces "
            "of information contributed most strongly to the "
            "model's decisions."
        ),
    )

    top_features = importance.head(12).copy()

    top_features["Feature"] = (
        top_features["feature"]
        .apply(clean_feature_name)
    )

    top_features["Importance"] = (
        top_features["importance"] * 100
    )

    top_features = top_features.sort_values(
        "Importance",
        ascending=True,
    )

    fig_features = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
    )

    fig_features.update_traces(
        marker_color="#176A47",
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig_features.update_layout(
        title="Top information used by the Random Forest",
        xaxis_title="Relative feature importance (%)",
        yaxis_title="",
        height=580,
        paper_bgcolor="#F6F8F3",
        plot_bgcolor="#F6F8F3",
    )

    st.plotly_chart(
        fig_features,
        use_container_width=True,
    )

    with st.container(border=True):

        st.subheader("What stands out?")

        st.write(
            """
            **Departure time was the strongest individual signal.**
            Time of day also contributed strongly.
            """
        )

        st.write(
            """
            **Weather contributed useful information.**
            Temperature, overall weather conditions, humidity,
            dew point and wind appear among the stronger features.
            """
        )

        st.write(
            """
            **Flight characteristics also mattered.**
            Airline, scheduled duration, distance and destination
            contributed to the model's predictions.
            """
        )

        st.caption(
            "Feature importance describes how the model used "
            "the data. It does not establish that a feature "
            "causes flight delays."
        )

    st.divider()

    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    section_intro(
        "Model Evaluation",
        "Which prediction approach performed better?",
        (
            "SkyPredict compared a Logistic Regression baseline "
            "with a Random Forest classifier."
        ),
    )

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
            ],
            "Logistic Regression": [
                logistic["accuracy"],
                logistic["precision"],
                logistic["recall"],
                logistic["f1"],
                logistic["roc_auc"],
            ],
            "Random Forest": [
                forest["accuracy"],
                forest["precision"],
                forest["recall"],
                forest["f1"],
                forest["roc_auc"],
            ],
        }
    )

    comparison_long = comparison.melt(
        id_vars="Metric",
        var_name="Model",
        value_name="Score",
    )

    fig_compare = px.bar(
        comparison_long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score",
        color_discrete_map={
            "Logistic Regression": "#A9CDB5",
            "Random Forest": "#176A47",
        },
    )

    fig_compare.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig_compare.update_layout(
        title="Logistic Regression vs. Random Forest",
        yaxis_title="Score",
        xaxis_title="",
        yaxis_range=[0, 0.85],
        height=480,
        paper_bgcolor="#F6F8F3",
        plot_bgcolor="#F6F8F3",
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True,
    )

    st.write(
        f"""
        The Random Forest produced the stronger overall results
        in this experiment, reaching an **F1 score of
        {forest['f1']:.3f}** and **ROC-AUC of
        {forest['roc_auc']:.3f}**.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # PERFORMANCE PLAIN ENGLISH
    # -----------------------------------------------------

    section_intro(
        "Testing",
        "How well did SkyPredict identify delayed flights?",
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Test Flights",
        f"{results['testing_rows']:,}"
    )

    p2.metric(
        "Delayed Flights Identified",
        f"{forest['recall'] * 100:.1f}%"
    )

    p3.metric(
        "ROC-AUC",
        f"{forest['roc_auc']:.3f}"
    )

    st.write(
        f"""
        On the held-out test data, the Random Forest identified
        approximately **{forest['recall'] * 100:.1f}% of the
        flights that were actually delayed**.
        """
    )

    st.write(
        """
        The system is therefore useful as a demonstration of
        an end-to-end predictive workflow, but its predictions
        should not be treated as operational airline forecasts.
        """
    )

    # -----------------------------------------------------
    # TECHNICAL DETAILS
    # -----------------------------------------------------

    st.divider()

    with st.expander(
        "Technical details for interested viewers"
    ):

        st.subheader("Model results")

        technical = comparison.copy()

        technical["Logistic Regression"] = (
            technical["Logistic Regression"]
            .map(lambda x: f"{x:.4f}")
        )

        technical["Random Forest"] = (
            technical["Random Forest"]
            .map(lambda x: f"{x:.4f}")
        )

        st.dataframe(
            technical,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Project stack")

        st.write(
            """
            Python · Pandas · scikit-learn · Random Forest ·
            Logistic Regression · Meteostat · Streamlit · Plotly
            """
        )

        st.subheader("Current scope")

        st.write(
            """
            The current model uses July 2026 flight records
            with historical weather observations for 20 major
            U.S. origin airports. Future work could add more
            seasons, additional airports and live weather forecasts.
            """
        )

    st.divider()

    if st.button(
        "Try SkyPredict Yourself",
        key="results_try",
    ):
        go_to("Try It Out")
        st.rerun()