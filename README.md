# 🏏 Cricket Win Predictor

A production-style starter project for predicting **T20 chase win probability** from ball-by-ball match state.

This repository is structured to help you go from raw IPL/T20 data to a trained model and a Streamlit app quickly.

---

## ✨ What this project does

- Ingests ball-by-ball cricket data (IPL / T20I).
- Engineers live chase-pressure features.
- Trains a machine learning pipeline (XGBoost or Logistic baseline).
- Serves predictions through a Streamlit web app.

---

## 📥 1) Data Prompt (Where to get data)

Use **Cricsheet** or **Kaggle** and download IPL/T20 ball-by-ball datasets (CSV/JSON).

If you're using search/API tools, use this exact prompt:

> **"Download the latest ball-by-ball cricket dataset for T20 matches including IPL 2025/2026. The data must include: match_id, venue, batting_team, bowling_team, ball_number, runs_off_bat, extras, and player_dismissed."**

### Minimum required raw columns

- `match_id`
- `venue`
- `batting_team`
- `bowling_team`
- `ball_number`
- `runs_off_bat`
- `extras`
- `player_dismissed`

---

## 🗂️ 2) Folder Structure

```plaintext
cricket_win_predictor/
│
├── data/                   # Raw and processed datasets
│   ├── raw_data.csv
│   └── final_df.csv        # Cleaned data for training
│
├── models/                 # Saved machine learning models
│   └── pipe.pkl            # Trained XGBoost/Logistic Pipeline
│
├── notebooks/              # Jupyter notebooks for testing
│   └── data_cleaning.ipynb
│
├── app/                    # Web application files
│   └── app.py              # Main Streamlit script
│
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

---

## 🧠 3) Technical Workflow Blueprint

To reach strong performance (targeting **80%+ accuracy**), your model must understand chase context—not just score.

### Feature engineering (Current State)

From each ball, compute:

- **Current Score**: cumulative runs by batting team up to current ball.
- **Runs Left**: `target - current_score`.
- **Balls Left**: `120 - balls_bowled`.
- **Wickets Left**: `10 - wickets_fallen`.
- **CRR (Current Run Rate)**: `current_score / overs_completed`.
- **RRR (Required Run Rate)**: `(runs_left / balls_left) * 6`.

### Suggested modeling flow

1. Load raw ball-by-ball data.
2. Build match-state rows ball-by-ball.
3. Keep only second-innings chase rows (if modeling chase win probability).
4. Split train/test by match (avoid leakage across balls in the same match).
5. Train baseline Logistic Regression pipeline.
6. Train XGBoost pipeline and compare AUC/accuracy/log-loss.
7. Save best model as `models/pipe.pkl`.

---

## 🚀 4) Streamlit App

The starter app is in `app/app.py` and includes:

- Team and city selectors.
- Target, score, overs, wickets inputs.
- Derived features (`runs_left`, `balls_left`, `wickets`, `crr`, `rrr`).
- Win probability display for batting and bowling teams.

### Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

> Ensure `models/pipe.pkl` exists before prediction.

---

## 🎯 5) How to make the model “Very Good”

- **Use XGBoost** for non-linear pressure interactions.
- **Apply recency weighting** (higher weight to last 2–3 years).
- **Prefer city over stadium name** to reduce category sparsity.
- Add richer context features:
  - phase (`powerplay`, `middle`, `death`)
  - recent momentum (last 6 balls runs/wickets)
  - venue-adjusted scoring trends
- Validate by season (time-based validation) to simulate real deployment.

---

## ✅ Next Steps Checklist

- [ ] Replace `data/raw_data.csv` with real ball-by-ball data.
- [ ] Build/extend `notebooks/data_cleaning.ipynb` to output `data/final_df.csv`.
- [ ] Train and serialize model to `models/pipe.pkl`.
- [ ] Launch Streamlit app and test multiple chase scenarios.

---

## 📦 Dependencies

Core libraries are listed in `requirements.txt`:

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Jupyter

---

## 📝 License

This project uses the repository's existing `LICENSE` file.
