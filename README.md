#here is an image of the product and the link is in the discription!
![Alt text]()

# 🏏 Cricket Win Predictor

A practical IPL/T20 prediction starter project with:

- **Data foundation** (match, player, venue, team-form, and ball-by-ball CSVs)
- **Feature-ready training frame** for chase prediction
- **Streamlit web app** to explore datasets and predict outcomes

---

## ✨ What this repository provides

- Sample IPL datasets across multiple granularities (match-level, player-level, and delivery-level)
- Starter chase-state training data (`final_df.csv`)
- Streamlit app with:
  - **Predict Winner** workflow
  - **Data Explorer** workflow for all CSVs in `data/`
  - **Heuristic fallback** prediction if `models/pipe.pkl` is not available

---

## 🗂️ Project structure

```plaintext
CRICKET_PREDICTOR/
│
├── app/
│   └── app.py
├── data/
│   ├── raw_data.csv
│   ├── final_df.csv
│   ├── player_stats.csv
│   ├── matches.csv
│   ├── deliveries.csv
│   ├── players.csv
│   ├── batting_stats.csv
│   ├── bowling_stats.csv
│   ├── venues.csv
│   ├── head_to_head.csv
│   ├── team_form.csv
│   ├── team_strength_metrics.csv
│   └── match_conditions.csv
├── notebooks/
│   └── data_cleaning.ipynb
├── models/
│   └── pipe.pkl              # optional (trained model)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset guide

### 1) Match-level data
- `data/matches.csv`
- Key fields include: `match_id`, `season`, `city`, `venue`, `team1`, `team2`, `toss_winner`, `winner`, `team1_score`, `team2_score`

### 2) Ball-by-ball data
- `data/deliveries.csv`
- Key fields include: `match_id`, `inning`, `over`, `ball`, `batsman`, `bowler`, `runs_scored`, `extras`, `wicket`

### 3) Player profile data
- `data/players.csv`, `data/player_stats.csv`
- Includes role, batting/bowling style, team, experience, and lifetime indicators

### 4) Player performance data
- `data/batting_stats.csv`
- `data/bowling_stats.csv`
- Includes strike-rate/economy and phase-relevant metrics

### 5) Team & venue context
- `data/venues.csv`
- `data/head_to_head.csv`
- `data/team_form.csv`
- `data/team_strength_metrics.csv`
- `data/match_conditions.csv`

These are useful for engineered features like:
- toss impact
- venue advantage
- recent form difference
- head-to-head ratio
- batting vs bowling strength gap

---

## 🧠 Modeling blueprint (recommended)

1. Start with `raw_data.csv` / `deliveries.csv`.
2. Build ball-by-ball chase-state features.
3. Use `final_df.csv` as a baseline schema.
4. Join context data (venue/team form/head-to-head/player strength).
5. Split train/test by match or season (avoid leakage).
6. Train baseline Logistic Regression and tree models (Random Forest / XGBoost).
7. Save the best pipeline to `models/pipe.pkl`.

---

## 🚀 Run the app

```bash
python scripts/train_model.py
```

### App capabilities

- **Predict Winner tab**
  - Enter batting team, bowling team, city, target, score, overs, wickets
  - See win probability for both teams
  - View derived match-state features + engineered context features

- **Data Explorer tab**
  - Browse all `data/*.csv`
  - Inspect rows/columns in-app
  - Download selected CSV directly

> If `models/pipe.pkl` does not exist, the app still predicts using a built-in heuristic fallback.

### App capabilities

- **Predict Winner tab**
  - Enter batting team, bowling team, city, target, score, overs, wickets
  - See win probability for both teams
  - View derived match-state features + engineered context features

## ✅ Next steps

- Replace sample CSVs with full historical IPL datasets.
- Add training scripts (`scripts/feature_engineering.py`, `scripts/train_model.py`).
- Evaluate with season-wise validation.
- Persist the best model to `models/pipe.pkl`.

---

## 📦 Core dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- jupyter

---

## 📝 License

This project uses the repository's `LICENSE` file.
