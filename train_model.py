import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Load Data
match = pd.read_csv('data/matches.csv')
delivery = pd.read_csv('data/deliveries.csv')

# 2. Get Total Score of 1st Innings
total_score_df = delivery.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]
match_df = match.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')

# 3. Clean Teams (Keep only consistent 2026 teams)
teams = ['Rajasthan Royals','Royal Challengers Bengaluru','Chennai Super Kings',
         'Delhi Capitals','Gujarat Titans','Kolkata Knight Riders',
         'Lucknow Super Giants','Mumbai Indians','Punjab Kings','Sunrisers Hyderabad']

match_df = match_df[match_df['team1'].isin(teams) & match_df['team2'].isin(teams)]
match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]
delivery_df = match_df.merge(delivery, on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 2]

# 4. Feature Engineering
delivery_df['current_score'] = delivery_df.groupby('match_id').cumsum()['total_runs_y']
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
delivery_df['balls_left'] = 120 - (delivery_df['over']*6 + delivery_df['ball'])
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: "0" if x == "0" else "1").astype('int')
wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets_left'] = 10 - wickets

delivery_df['crr'] = (delivery_df['current_score']*6) / (120 - delivery_df['balls_left'])
delivery_df['rrr'] = (delivery_df['runs_left']*6) / delivery_df['balls_left']

def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0

delivery_df['result'] = delivery_df.apply(result, axis=1)

# 5. Final Extraction
final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets_left','total_runs_x','crr','rrr','result']]
final_df = final_df.sample(final_df.shape[0]) # Shuffle
final_df.dropna(inplace=True)
final_df = final_df[final_df['balls_left'] != 0]

# 6. Model Building
X = final_df.iloc[:, :-1]
y = final_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse=False, drop='first'), ['batting_team','bowling_team','city'])
], remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train, y_train)

# 7. Save the Model
pickle.dump(pipe, open('pipe.pkl', 'wb'))
print("Model trained and saved as pipe.pkl!")
