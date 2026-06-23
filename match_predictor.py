import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- 1. Load data ----------
# Expect one row per historical match with raw stats for team_a and team_b
df = pd.read_csv("matches.csv")

# ---------- 2. Feature engineering (diffs) ----------
feature_cols = ["gdp_per_capita", "population", "squad_value", "fifa_rank"]  # adjust to actual columns

for col in feature_cols:
    df[f"{col}_diff"] = df[f"team_a_{col}"] - df[f"team_b_{col}"]

diff_cols = [f"{col}_diff" for col in feature_cols]

# ---------- 3. Symmetry augmentation ----------
# Mirror every match: swap A/B, flip diffs, flip label
mirrored = df.copy()
mirrored[diff_cols] = -mirrored[diff_cols]
mirrored["label"] = mirrored["label"].map({0: 1, 1: 0, 2: 2})  # draw stays draw

full_df = pd.concat([df, mirrored], ignore_index=True)

X = full_df[diff_cols]
y = full_df["label"]

# ---------- 4. Train/test split (BEFORE scaling) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 5. Scale ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 6. Model ----------
# Class labels: 0 = team_b wins, 1 = team_a wins, 2 = draw
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

# ---------- 7. Evaluate ----------
target_names = ["Team B win", "Team A win", "Draw"]  # adjust if your label encoding differs

preds = rf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds, target_names=target_names))

# ---------- 8. Inference function ----------
def predict_match(team_a_stats: dict, team_b_stats: dict, model=rf):
    """
    team_a_stats / team_b_stats: dict with keys matching feature_cols
    Returns class probabilities [P(Team B wins), P(Team A wins), P(Draw)]
    """
    diffs = [[team_a_stats[col] - team_b_stats[col] for col in feature_cols]]
    diffs_scaled = scaler.transform(diffs)
    probs = model.predict_proba(diffs_scaled)[0]
    return dict(zip(target_names, probs))