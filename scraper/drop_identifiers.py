"""
drop_identifiers.py
===================
Make a model-ready copy of the trainable match file with identity columns removed.

Drops:
  home_team, away_team  -> the model should learn from features, not team names
  year                  -> only used to join features on; the date column still
                           carries the temporal info if it's needed later

Reads : ../data/matches_with_features_trainable.csv
Writes: ../data/matches_with_features_model.csv
"""

import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "../data")

IN_FILE = os.path.join(DATA, "matches_with_features_trainable.csv")
OUT_FILE = os.path.join(DATA, "matches_with_features_model.csv")

DROP_COLS = ["home_team", "away_team", "year"]


def main():
    df = pd.read_csv(IN_FILE)
    print(f"input : {IN_FILE}")
    print(f"        {df.shape[0]} rows x {df.shape[1]} cols")

    # only drop what's actually present, so a re-run or a renamed column won't crash it
    present = [c for c in DROP_COLS if c in df.columns]
    missing = [c for c in DROP_COLS if c not in df.columns]
    if missing:
        print(f"        note: not found, skipping -> {missing}")

    out = df.drop(columns=present)

    out.to_csv(OUT_FILE, index=False)
    print(f"dropped: {present}")
    print(f"output: {OUT_FILE}")
    print(f"        {out.shape[0]} rows x {out.shape[1]} cols")
    print(f"        columns: {list(out.columns)}")


if __name__ == "__main__":
    main()