import math
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cricket Data & Win Predictor", page_icon="🏏", layout="wide")

TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
]

CITIES = ["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Dubai", "Hyderabad", "Jaipur", "Kolkata", "Mumbai", "Pune"]
MODEL_PATH = Path("models/pipe.pkl")
DATA_DIR = Path("data")


@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        return None
    with model_path.open("rb") as f:
        return pickle.load(f)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def list_data_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.csv"))


def validate_inputs(target: int, score: int, overs: float, wickets_fallen: int) -> bool:
    if target <= 0:
        st.error("Target score must be greater than 0.")
        return False
    if score < 0:
        st.error("Current score cannot be negative.")
        return False
    if not 0 <= overs <= 20:
        st.error("Overs completed must be between 0 and 20.")
        return False
    if not 0 <= wickets_fallen <= 10:
        st.error("Wickets fallen must be between 0 and 10.")
        return False
    return True


def heuristic_win_probability(runs_left: int, balls_left: int, wickets_left: int) -> float:
    if runs_left <= 0:
        return 0.99
    if balls_left <= 0:
        return 0.01
    required_rr = (runs_left * 6) / balls_left
    pressure = 0.85 * required_rr - 0.22 * wickets_left
    return 1 / (1 + math.exp(pressure - 6.0))


def build_context_features(batting_team: str, bowling_team: str, city: str) -> pd.DataFrame:
    team_form = load_csv(DATA_DIR / "team_form.csv")
    venues = load_csv(DATA_DIR / "venues.csv")
    h2h = load_csv(DATA_DIR / "head_to_head.csv")

    row = {
        "team1_win_rate_last_10": None,
        "team2_win_rate_last_10": None,
        "venue_win_rate": None,
        "head_to_head_ratio": None,
        "recent_form_difference": None,
        "batting_strength_difference": None,
        "bowling_strength_difference": None,
        "toss_win_advantage": None,
    }

    if not team_form.empty:
        t1 = team_form[team_form["team"] == batting_team]
        t2 = team_form[team_form["team"] == bowling_team]
        if not t1.empty:
            row["team1_win_rate_last_10"] = round((t1.iloc[0]["last_5_matches_wins"] / 5) * 100, 2)
        if not t2.empty:
            row["team2_win_rate_last_10"] = round((t2.iloc[0]["last_5_matches_wins"] / 5) * 100, 2)
        if not t1.empty and not t2.empty:
            row["recent_form_difference"] = round(
                t1.iloc[0]["last_5_matches_wins"] - t2.iloc[0]["last_5_matches_wins"], 2
            )
            row["batting_strength_difference"] = round(
                t1.iloc[0]["batting_strength"] - t2.iloc[0]["batting_strength"], 2
            )
            row["bowling_strength_difference"] = round(
                t1.iloc[0]["bowling_strength"] - t2.iloc[0]["bowling_strength"], 2
            )

    if not venues.empty:
        venue_row = venues[venues["city"] == city]
        if not venue_row.empty:
            row["venue_win_rate"] = float(venue_row.iloc[0]["chasing_win_rate"])

    if not h2h.empty:
        pair = h2h[(h2h["team1"] == batting_team) & (h2h["team2"] == bowling_team)]
        reverse = h2h[(h2h["team1"] == bowling_team) & (h2h["team2"] == batting_team)]
        if not pair.empty:
            p = pair.iloc[0]
            row["head_to_head_ratio"] = round(p["team1_wins"] / max(p["team2_wins"], 1), 3)
        elif not reverse.empty:
            r = reverse.iloc[0]
            row["head_to_head_ratio"] = round(r["team2_wins"] / max(r["team1_wins"], 1), 3)

    row["toss_win_advantage"] = 3.5
    return pd.DataFrame([row])


st.title("🏏 IPL Data Explorer & Win Predictor")
st.caption("Access IPL datasets and predict T20 chase outcomes.")

model = load_model(MODEL_PATH)
if model is None:
    st.info("`models/pipe.pkl` not found. Using heuristic fallback for predictions.")
else:
    st.success("Loaded trained model from `models/pipe.pkl`.")

predict_tab, data_tab = st.tabs(["Predict Winner", "Data Explorer"])

with predict_tab:
    c1, c2, c3 = st.columns(3)
    batting_team = c1.selectbox("Batting Team", TEAMS)
    bowling_team = c2.selectbox("Bowling Team", [t for t in TEAMS if t != batting_team])
    city = c3.selectbox("City", CITIES)

    c4, c5, c6, c7 = st.columns(4)
    target = c4.number_input("Target", min_value=1, max_value=300, value=180, step=1)
    score = c5.number_input("Current Score", min_value=0, max_value=350, value=100, step=1)
    overs = c6.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
    wickets_fallen = c7.number_input("Wickets Fallen", min_value=0, max_value=10, value=3, step=1)

    if st.button("Predict", type="primary"):
        if validate_inputs(target, score, overs, wickets_fallen):
            balls_bowled = int(round(overs * 6))
            runs_left = max(target - score, 0)
            balls_left = max(120 - balls_bowled, 0)
            wickets_left = 10 - wickets_fallen
            crr = score / overs if overs > 0 else 0.0
            rrr = (runs_left * 6) / max(balls_left, 1)

            model_input = pd.DataFrame(
                {
                    "batting_team": [batting_team],
                    "bowling_team": [bowling_team],
                    "city": [city],
                    "runs_left": [runs_left],
                    "balls_left": [max(balls_left, 1)],
                    "wickets": [wickets_left],
                    "total_runs_x": [target],
                    "crr": [crr],
                    "rrr": [rrr],
                }
            )

            if model is not None:
                prob = float(model.predict_proba(model_input)[0][1])
            else:
                prob = heuristic_win_probability(runs_left, balls_left, wickets_left)

            st.subheader("Prediction")
            m1, m2 = st.columns(2)
            m1.metric(batting_team, f"{prob * 100:.1f}%")
            m2.metric(bowling_team, f"{(1 - prob) * 100:.1f}%")

            st.markdown("### Match-State Features")
            st.dataframe(model_input, use_container_width=True, hide_index=True)

            st.markdown("### Engineered Context Features")
            st.dataframe(
                build_context_features(batting_team, bowling_team, city),
                use_container_width=True,
                hide_index=True,
            )

with data_tab:
    st.subheader("Available CSV Datasets")
    csv_files = list_data_files(DATA_DIR)
    if not csv_files:
        st.warning("No CSV files found in `data/`.")
    else:
        selected = st.selectbox("Select dataset", [p.name for p in csv_files])
        selected_path = DATA_DIR / selected
        df = load_csv(selected_path)
        st.caption(f"Path: `{selected_path}`")
        st.write(f"Rows: **{len(df)}** | Columns: **{len(df.columns)}**")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name=selected,
            mime="text/csv",
        )
