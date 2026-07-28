import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. Load & prep data
# ============================================================
df = pd.read_csv("matches_with_features_model.csv")
label_map = {"H": 1, "A": 0, "D": 2}
df["label"] = df["result"].map(label_map)

diff_cols = ["elo_diff", "age_mean_diff", "age_std_diff", "value_mean_diff", "value_std_diff"]

# Symmetry augmentation: mirror every match (swap home/away, flip diffs, flip label; draw stays draw)
mirrored = df.copy()
mirrored[diff_cols] = -mirrored[diff_cols]
mirrored["label"] = mirrored["label"].map({0: 1, 1: 0, 2: 2})
full_df = pd.concat([df, mirrored], ignore_index=True)

# ============================================================
# 2. PART A: Single 3-class models -- Logistic Regression vs Random Forest
# ============================================================
X = full_df[diff_cols]
y = full_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler_3class = StandardScaler()
X_train_scaled = scaler_3class.fit_transform(X_train)
X_test_scaled = scaler_3class.transform(X_test)

target_names_3class = ["Away win", "Home win", "Draw"]

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

rf_3class = RandomForestClassifier(n_estimators=200, random_state=42)
rf_3class.fit(X_train_scaled, y_train)

print("=" * 60)
print("PART A: Single 3-class model comparison")
print("=" * 60)
for name, model in [("Logistic Regression", log_reg), ("Random Forest", rf_3class)]:
    preds = model.predict(X_test_scaled)
    print(f"\n--- {name} ---")
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print(classification_report(y_test, preds, target_names=target_names_3class, zero_division=0))

# ============================================================
# 3. PART B: Split models -- Draw (binary) + Win/Loss (binary)
# ============================================================
full_df["is_draw"] = (full_df["label"] == 2).astype(int)
X_draw = full_df[diff_cols]
y_draw = full_df["is_draw"]

Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_draw, y_draw, test_size=0.2, random_state=42, stratify=y_draw
)
scaler_draw = StandardScaler()
Xd_train_s = scaler_draw.fit_transform(Xd_train)
Xd_test_s = scaler_draw.transform(Xd_test)

draw_model = RandomForestClassifier(n_estimators=200, random_state=42)
draw_model.fit(Xd_train_s, yd_train)
draw_preds = draw_model.predict(Xd_test_s)

nondraw_df = full_df[full_df["label"] != 2].copy()
X_wl = nondraw_df[diff_cols]
y_wl = nondraw_df["label"]

Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    X_wl, y_wl, test_size=0.2, random_state=42, stratify=y_wl
)
scaler_wl = StandardScaler()
Xw_train_s = scaler_wl.fit_transform(Xw_train)
Xw_test_s = scaler_wl.transform(Xw_test)

wl_model = RandomForestClassifier(n_estimators=200, random_state=42)
wl_model.fit(Xw_train_s, yw_train)
wl_preds = wl_model.predict(Xw_test_s)

print("\n" + "=" * 60)
print("PART B: Split models (Draw binary + Win/Loss binary)")
print("=" * 60)

print("\n--- Draw Model (draw vs not-draw) ---")
print("Accuracy:", round(accuracy_score(yd_test, draw_preds), 4))
print(classification_report(yd_test, draw_preds, target_names=["Not draw", "Draw"], zero_division=0))

print("\n--- Win/Loss Model (non-draw matches only) ---")
print("Accuracy:", round(accuracy_score(yw_test, wl_preds), 4))
print(classification_report(yw_test, wl_preds, target_names=["Away win", "Home win"], zero_division=0))

# ============================================================
# 4. PART C: Real-world inference -- Argentina 2022 vs Germany 2014
# ============================================================
def predict_matchup(team_a_name, team_a_stats, team_b_name, team_b_stats):
    row = {
        "elo_diff": team_a_stats["elo"] - team_b_stats["elo"],
        "age_mean_diff": team_a_stats["age_mean"] - team_b_stats["age_mean"],
        "age_std_diff": team_a_stats["age_std"] - team_b_stats["age_std"],
        "value_mean_diff": team_a_stats["value_mean"] - team_b_stats["value_mean"],
        "value_std_diff": team_a_stats["value_std"] - team_b_stats["value_std"],
    }
    X_new = pd.DataFrame([row])[diff_cols]
    print(f"\n========== {team_a_name} vs {team_b_name} ==========")
    print("Raw diffs:", row)

    X_new_3class = scaler_3class.transform(X_new)
    for name, model in [("Logistic Regression", log_reg), ("Random Forest", rf_3class)]:
        probs = model.predict_proba(X_new_3class)[0]
        print(f"\n[{name}] (single 3-class model)")
        for cls, p in zip(target_names_3class, probs):
            label = cls.replace("Home win", f"{team_a_name} win").replace("Away win", f"{team_b_name} win")
            print(f"  {label}: {p*100:.1f}%")

    X_new_draw = scaler_draw.transform(X_new)
    draw_prob = draw_model.predict_proba(X_new_draw)[0][1]
    X_new_wl = scaler_wl.transform(X_new)
    wl_probs = wl_model.predict_proba(X_new_wl)[0]
    print(f"\n[Split models: Draw model + Win/Loss model]")
    print(f"  Draw probability: {draw_prob*100:.1f}%")
    print(f"  If not a draw -> {team_a_name} win: {wl_probs[1]*100:.1f}% | {team_b_name} win: {wl_probs[0]*100:.1f}%")


argentina_2022 = {
    "elo": 2144,
    "age_mean": 28.81,
    "age_std": 4.105,
    "value_mean": 28_769_231,
    "value_std": 23_981_255,
}

germany_2014 = {
    "elo": 2133,
    "age_mean": 27.0,
    "age_std": 3.683,
    "value_mean": 24_608_696,
    "value_std": 15_905_944,
}

print("\n" + "=" * 60)
print("PART C: Real-world inference")
print("=" * 60)
predict_matchup("Argentina 2022", argentina_2022, "Germany 2014", germany_2014)