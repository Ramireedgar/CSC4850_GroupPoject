import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

print("Starting data extraction and scaling...")

# Connect to database
db_path = 'database.sqlite'
if not os.path.exists(db_path):
    db_path = os.path.join('pre-processing', 'database.sqlite')

conn = sqlite3.connect(db_path)

# Query matches
query = """
    SELECT 
        m.id, 
        m.date,
        ht.team_long_name AS home_team,
        at.team_long_name AS away_team,
        m.home_team_goal, 
        m.away_team_goal,
        m.home_team_api_id,
        m.away_team_api_id
    FROM Match m
    LEFT JOIN Team ht ON m.home_team_api_id = ht.team_api_id
    LEFT JOIN Team at ON m.away_team_api_id = at.team_api_id
"""
matches = pd.read_sql_query(query, conn)

# Outcome mapping
def get_outcome(row):
    if row['home_team_goal'] > row['away_team_goal']:
        return 1 # Win
    elif row['home_team_goal'] < row['away_team_goal']:
        return 2 # Loss
    else:
        return 0 # Draw
matches['match_outcome'] = matches.apply(get_outcome, axis=1)

# Team Attributes
team_query = "SELECT * FROM Team_Attributes"
team_attributes = pd.read_sql_query(team_query, conn)
numeric_team_cols = team_attributes.select_dtypes(include=[np.number]).columns
team_stats_avg = team_attributes[numeric_team_cols].groupby('team_api_id').mean().reset_index()
team_stats_avg = team_stats_avg.drop(columns=['id', 'team_fifa_api_id'], errors='ignore')

matches = pd.merge(matches, team_stats_avg.add_prefix('home_'), 
                   left_on=['home_team_api_id'], right_on=['home_team_api_id'], how='left')
matches = pd.merge(matches, team_stats_avg.add_prefix('away_'), 
                   left_on=['away_team_api_id'], right_on=['away_team_api_id'], how='left')

# Symmetrization
df1 = matches.copy()
df1['is_home'] = 1

df2 = matches.copy()
df2['is_home'] = 0
df2['match_outcome'] = df2['match_outcome'].map({1: 2, 2: 1, 0: 0})

home_cols = [c for c in matches.columns if c.startswith('home_')]
away_cols = [c for c in matches.columns if c.startswith('away_')]
rename_df1 = {c: c.replace('home_', 'team_') for c in home_cols}
rename_df1.update({c: c.replace('away_', 'opponent_') for c in away_cols})
rename_df2 = {c: c.replace('away_', 'team_') for c in away_cols}
rename_df2.update({c: c.replace('home_', 'opponent_') for c in home_cols})

df1 = df1.rename(columns=rename_df1)
df2 = df2.rename(columns=rename_df2)
matches = pd.concat([df1, df2], ignore_index=True)

# Sort by date for temporal split
matches = matches.sort_values(by='date').reset_index(drop=True)

# Keep numerical only
numeric_cols = matches.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = matches[numeric_cols].copy()

# Drop IDs
cols_to_drop = ['id', 'team_team_goal', 'opponent_team_goal', 'team_team_api_id', 'opponent_team_api_id']
cols_to_drop.extend([col for col in df_numeric.columns if 'id' in col]) 
df_numeric = df_numeric.drop(columns=[c for c in cols_to_drop if c in df_numeric.columns], errors='ignore')

# Split features/target
X = df_numeric.drop('match_outcome', axis=1)
y = df_numeric['match_outcome']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Impute median based on train
train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test  = X_test.fillna(train_median)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Convert back to DataFrame to preserve column names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Export
out_dir = 'pre-processing' if not db_path.startswith('pre-processing') else '.'
X_train_scaled_df.to_csv(os.path.join(out_dir, 'X_train_scaled.csv'), index=False)
X_test_scaled_df.to_csv(os.path.join(out_dir, 'X_test_scaled.csv'), index=False)

print(f"Successfully exported scaled (un-reduced) data with {X_train_scaled_df.shape[1]} features.")
