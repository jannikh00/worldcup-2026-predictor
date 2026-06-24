"""
join_features.py
================
Pipeline Step 3 — attach team features to each historical match.

For every row in matches_with_elo.csv we look up the squad features of BOTH
teams *for the year the match was played in*, then add:

  home_age_mean / away_age_mean        age_mean_diff   (= home - away)
  home_age_std  / away_age_std         age_std_diff
  home_value_mean / away_value_mean    value_mean_diff
  home_value_std  / away_value_std     value_std_diff

Why home_/away_ and not just "age_mean":
  A match has two teams, so each base feature needs a home and an away copy.
  The four *_diff columns are the per-feature comparison the prediction model
  actually trains on. Diffs are computed home - away, matching the direction of
  your existing elo_diff (= elo_home_pre - elo_away_pre), so all "_diff" columns
  in the file point the same way.

Year matching:
  The match year comes from the date column; each team is joined on
  (team, that_year). So a 2022 fixture uses 2022 squads, a 2026 fixture uses
  2026 squads, etc. The feature table covers 2022-2026, which spans every match.

Coverage:
  The match file contains hundreds of national teams; the feature table only
  has the 48 2026 World Cup squads. Matches involving a non-feature team keep
  the row but get NaN for that side's features (and NaN diffs). A second file,
  matches_with_features_trainable.csv, is written containing only rows where
  BOTH teams were found - that's the directly trainable subset.

Run:  python3 join_features.py
"""

import os
import pandas as pd

_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FEATURES = os.path.join(_DIR, "team_features_rescraped.csv")
MATCHES  = os.path.join(_DIR, "matches_with_elo.csv")
OUT_ALL       = os.path.join(_DIR, "matches_with_features.csv")
OUT_TRAINABLE = os.path.join(_DIR, "matches_with_features_trainable.csv")

# The four per-team features we attach (n_players is left out of the join on
# purpose; add it here if you want it as a feature too).
BASE = ["age_mean", "age_std", "value_mean", "value_std"]


def main():
    feat = pd.read_csv(FEATURES)
    matches = pd.read_csv(MATCHES)

    # --- normalise the join keys -----------------------------------------
    feat["country"] = feat["country"].astype(str).str.strip()
    matches["home_team"] = matches["home_team"].astype(str).str.strip()
    matches["away_team"] = matches["away_team"].astype(str).str.strip()

    # match year drives which squad snapshot we use
    matches["year"] = pd.to_datetime(matches["date"]).dt.year

    feat_small = feat[["country", "year"] + BASE].copy()

    # --- join the home side ----------------------------------------------
    home = feat_small.rename(
        columns={"country": "home_team", **{c: f"home_{c}" for c in BASE}})
    matches = matches.merge(home, on=["home_team", "year"], how="left")

    # --- join the away side ----------------------------------------------
    away = feat_small.rename(
        columns={"country": "away_team", **{c: f"away_{c}" for c in BASE}})
    matches = matches.merge(away, on=["away_team", "year"], how="left")

    # --- differences (home - away) ---------------------------------------
    for c in BASE:
        matches[f"{c}_diff"] = matches[f"home_{c}"] - matches[f"away_{c}"]

    matches.to_csv(OUT_ALL, index=False)

    # --- trainable subset: both teams resolved ---------------------------
    both = matches["home_age_mean"].notna() & matches["away_age_mean"].notna()
    matches[both].to_csv(OUT_TRAINABLE, index=False)

    # --- report ----------------------------------------------------------
    home_ok = matches["home_age_mean"].notna()
    away_ok = matches["away_age_mean"].notna()
    n = len(matches)
    print(f"Matches in            : {n}")
    print(f"Both teams matched    : {both.sum()}  -> {OUT_TRAINABLE}")
    print(f"Only home matched     : {(home_ok & ~away_ok).sum()}")
    print(f"Only away matched     : {(~home_ok & away_ok).sum()}")
    print(f"Neither matched       : {(~home_ok & ~away_ok).sum()}")
    print(f"Full file (all rows)  : {OUT_ALL}")

    # Sanity check: any feature-table country that NEVER appears in matches
    # under the team-name spelling we joined on? That would signal a naming
    # mismatch between the two files rather than a genuinely absent team.
    match_names = set(matches["home_team"]) | set(matches["away_team"])
    missing = sorted(set(feat["country"]) - match_names)
    if missing:
        print("\nFeature-table teams that never appear in matches "
              "(check spelling if unexpected):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()