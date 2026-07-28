"""
join_features.py
================
attach team features to each historical match.

For every row in matches_with_elo.csv we look up the squad features of BOTH
teams *for the year the match was played in*, then add:

  home_age_mean / away_age_mean        age_mean_diff   (= home - away)
  home_age_std  / away_age_std         age_std_diff
  home_value_mean / away_value_mean    value_mean_diff
  home_value_std  / away_value_std     value_std_diff

Year matching:
  The match year comes from the date column; each team is joined on
  (team, that_year). So a 2022 fixture uses 2022 squads, a 2026 fixture uses
  2026 squads, etc. The feature table covers 2022-2026, which spans every match.

Coverage:
  The match file contains hundreds of national teams; the feature table only
  has the 48 2026 World Cup squads. Matches involving a non-feature team keep
  the row but get NaN for that side's features (and NaN diffs). The subset
  where BOTH teams resolved is what make_model_table.py turns into the
  model-ready table.
"""

import pandas as pd

from paths import TEAM_FEATURES_CSV, MATCHES_ELO_CSV, MATCHES_FEATURES_CSV, ensure_dirs

FEATURES = TEAM_FEATURES_CSV
MATCHES = MATCHES_ELO_CSV
OUT_ALL = MATCHES_FEATURES_CSV

# The four per-team features we attach
BASE = ["age_mean", "age_std", "value_mean", "value_std"]


def main():
    ensure_dirs()
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

    # --- report ----------------------------------------------------------
    home_ok = matches["home_age_mean"].notna()
    away_ok = matches["away_age_mean"].notna()
    both = home_ok & away_ok
    n = len(matches)
    print(f"Matches in            : {n}")
    print(f"Both teams matched    : {both.sum()}  (-> make_model_table.py)")
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