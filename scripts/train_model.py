import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path("data")
MODEL_PATH = Path("models/pipe.pkl")


def build_training_frame(match_path: Path, delivery_path: Path) -> pd.DataFrame:
    match = pd.read_csv(match_path)
    delivery = pd.read_csv(delivery_path)

    # Normalize delivery columns expected by the training pipeline.
    delivery["total_runs"] = delivery["runs_scored"] + delivery["extras"]
    delivery["player_dismissed"] = delivery["wicket"].fillna(0).astype(int)

    # Use match-level first-innings score as chase target for second innings rows.
    match_df = match.copy()
    match_df["first_innings_score"] = match_df["team1_score"]

    teams = [
        "Rajasthan Royals",
        "Royal Challengers Bangalore",
        "Chennai Super Kings",
        "Delhi Capitals",
        "Gujarat Titans",
        "Kolkata Knight Riders",
        "Lucknow Super Giants",
        "Mumbai Indians",
        "Punjab Kings",
        "Sunrisers Hyderabad",
    ]

    match_df = match_df[match_df["team1"].isin(teams) & match_df["team2"].isin(teams)].copy()

    # Derive batting/bowling teams from inning in this sample schema.
    delivery_df = match_df.merge(delivery, on="match_id", how="inner")
    delivery_df = delivery_df[delivery_df["inning"] == 2].copy()

    delivery_df["batting_team"] = delivery_df["team2"]
    delivery_df["bowling_team"] = delivery_df["team1"]

    delivery_df["current_score"] = delivery_df.groupby("match_id")["total_runs"].cumsum()
    delivery_df["runs_left"] = delivery_df["first_innings_score"] - delivery_df["current_score"]
    balls_bowled = (delivery_df["over"] - 1) * 6 + delivery_df["ball"]
    delivery_df["balls_left"] = 120 - balls_bowled

    wickets_fallen = delivery_df.groupby("match_id")["player_dismissed"].cumsum()
    delivery_df["wickets"] = 10 - wickets_fallen

    balls_used = 120 - delivery_df["balls_left"]
    delivery_df["crr"] = (delivery_df["current_score"] * 6) / balls_used.clip(lower=1)
    delivery_df["rrr"] = (delivery_df["runs_left"] * 6) / delivery_df["balls_left"].clip(lower=1)
    delivery_df["result"] = (delivery_df["batting_team"] == delivery_df["winner"]).astype(int)

    final_df = delivery_df[
        [
            "batting_team",
            "bowling_team",
            "city",
            "runs_left",
            "balls_left",
            "wickets",
            "first_innings_score",
            "crr",
            "rrr",
            "result",
        ]
    ].rename(columns={"first_innings_score": "total_runs_x"})

    final_df = final_df.dropna().copy()
    final_df = final_df[(final_df["balls_left"] > 0) & (final_df["wickets"] >= 0)]

    return final_df


def train_and_save(df: pd.DataFrame, output_path: Path) -> None:
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if y.nunique() < 2:
        raise ValueError("Training data must include both classes in 'result'.")

    if len(df) >= 10 and y.value_counts().min() >= 2:
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=1, stratify=y
        )
    else:
        X_train, y_train = X, y

    trf = ColumnTransformer(
        [
            (
                "trf",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first"),
                ["batting_team", "bowling_team", "city"],
            )
        ],
        remainder="passthrough",
    )

    pipe = Pipeline(
        steps=[
            ("step1", trf),
            ("step2", LogisticRegression(solver="liblinear", max_iter=200)),
        ]
    )

    pipe.fit(X_train, y_train)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(pipe, f)


if __name__ == "__main__":
    training_df = build_training_frame(DATA_DIR / "matches.csv", DATA_DIR / "deliveries.csv")
    train_and_save(training_df, MODEL_PATH)
    print(f"Model trained and saved: {MODEL_PATH}")
