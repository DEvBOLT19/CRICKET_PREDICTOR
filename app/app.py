import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Live Cricket Win Predictor", page_icon="🏏", layout="centered")

TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Kolkata Knight Riders",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants",
]

CITIES = [
    "Ahmedabad",
    "Bengaluru",
    "Chennai",
    "Delhi",
    "Dubai",
    "Hyderabad",
    "Jaipur",
    "Kolkata",
    "Mumbai",
    "Pune",
]

MODEL_PATH = Path("models/pipe.pkl")


@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        return None
    with model_path.open("rb") as f:
        return pickle.load(f)


def validate_inputs(target: int, score: int, overs: float, wickets: int):
    if target <= 0:
        st.error("Target score must be greater than 0.")
        return False
    if score < 0:
        st.error("Current score cannot be negative.")
        return False
    if overs < 0 or overs > 20:
        st.error("Overs completed must be between 0 and 20.")
        return False
    if wickets < 0 or wickets > 10:
        st.error("Wickets fallen must be between 0 and 10.")
        return False
    if score > target:
        st.warning("Current score is already above target. Adjust inputs for a chase scenario.")
    return True


st.title("🏏 Live Cricket Win Predictor")
st.caption("Predict win probability in a T20 chase using engineered match-state features.")

model = load_model(MODEL_PATH)

if model is None:
    st.warning("Model file not found at `models/pipe.pkl`. Train and save a pipeline before predicting.")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Select Batting Team", TEAMS)
with col2:
    bowling_team = st.selectbox("Select Bowling Team", [t for t in TEAMS if t != batting_team])

city = st.selectbox("Select Host City", CITIES)
target = st.number_input("Target Score", min_value=1, max_value=300, value=180, step=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input("Current Score", min_value=0, max_value=350, value=100, step=1)
with col4:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
with col5:
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=3, step=1)

if st.button("Predict Probability", type="primary"):
    if validate_inputs(target, score, overs, wickets) and model is not None:
        balls_bowled = int(round(overs * 6))
        runs_left = max(target - score, 0)
        balls_left = max(120 - balls_bowled, 1)
        wickets_left = 10 - wickets

        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame(
            {
                "batting_team": [batting_team],
                "bowling_team": [bowling_team],
                "city": [city],
                "runs_left": [runs_left],
                "balls_left": [balls_left],
                "wickets": [wickets_left],
                "total_runs_x": [target],
                "crr": [crr],
                "rrr": [rrr],
            }
        )

        result = model.predict_proba(input_df)

        st.subheader("Win Probability")
        st.metric(label=batting_team, value=f"{round(result[0][1] * 100)}%")
        st.metric(label=bowling_team, value=f"{round(result[0][0] * 100)}%")
