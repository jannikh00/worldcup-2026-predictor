"""
make_model_table.py
===================
Turn the fully joined match file into the model-ready table.

Three things happen here:

1. ROW FILTER - keep only matches where BOTH teams have squad features.
   The feature table covers the 48 nations qualified for 2026; matches
   involving anyone else carry NaN features and cannot be trained on.

2. LEAKAGE GUARD - drop `home_score` / `away_score`.
   The target `result` is a deterministic function of those two columns
   (H / A / D), so leaving them in a table labelled "model-ready" invites a
   model that scores ~100% by reading the answer. They stay available in
   matches_with_features.csv for analysis; they just don't ship here.

3. IDENTITY DROP - drop `home_team` / `away_team` / `year`.
   The model should learn from ratings and squad composition, not from
   memorising which nations tend to win. `date` is kept: it carries the
   temporal information a time-aware split needs.

Reads : data/processed/matches_with_features.csv
Writes: data/processed/matches_model.csv
"""

import pandas as pd

from paths import MATCHES_FEATURES_CSV, MATCHES_MODEL_CSV, ensure_dirs

IN_FILE = MATCHES_FEATURES_CSV
OUT_FILE = MATCHES_MODEL_CSV

# Columns the target is derived from - dropping these prevents target leakage.
LEAKY_COLS = ["home_score", "away_score"]

# Identity columns the model should not key off.
IDENTITY_COLS = ["home_team", "away_team", "year"]


def main():
    ensure_dirs()
    df = pd.read_csv(IN_FILE)
    print(f"input : {IN_FILE}")
    print(f"        {df.shape[0]} rows x {df.shape[1]} cols")

    # 1. keep only fully-featured matches
    both = df["home_age_mean"].notna() & df["away_age_mean"].notna()
    df = df[both].copy()
    print(f"kept  : {both.sum()} rows where both teams have squad features")

    # 2 + 3. drop leaky and identity columns (only those actually present, so a
    # re-run or a renamed column won't crash the script)
    to_drop = LEAKY_COLS + IDENTITY_COLS
    present = [c for c in to_drop if c in df.columns]
    missing = [c for c in to_drop if c not in df.columns]
    if missing:
        print(f"        note: not found, skipping -> {missing}")

    out = df.drop(columns=present)

    # Sort chronologically so the file itself is in time order - makes the
    # time-based split in models/ obvious and reproducible.
    # kind="mergesort" is deliberate: pandas defaults to quicksort, which is
    # NOT stable. Dozens of matches share a date, so an unstable sort permutes
    # them arbitrarily and any downstream script that re-sorts would land on a
    # different row order - and therefore a different random train/test split.
    out = out.sort_values("date", kind="mergesort").reset_index(drop=True)

    out.to_csv(OUT_FILE, index=False)
    print(f"dropped: {present}")
    print(f"output: {OUT_FILE}")
    print(f"        {out.shape[0]} rows x {out.shape[1]} cols")
    print(f"        {out['date'].min()} -> {out['date'].max()}")
    print(f"        columns: {list(out.columns)}")


if __name__ == "__main__":
    main()
