"""
preview_heads.py
================
Print the head of the two joined match files for a quick side-by-side look.

Files live in the neighbouring data/ folder:
  ../data/matches_with_features.csv             (all matches, NaN where unmatched)
  ../data/matches_with_features_trainable.csv   (only matches with both teams resolved)

Run:  python3 preview_heads.py
"""

import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "../data")

FILES = [
    "matches_with_features.csv",
    "matches_with_features_trainable.csv",
]

N = 10  # rows to show

# show every column, don't truncate the middle, keep it on wide lines
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 18)


def main():
    for name in FILES:
        path = os.path.join(DATA, name)
        print("=" * 100)
        print(name)
        print("=" * 100)
        if not os.path.exists(path):
            print(f"  (not found at {path})\n")
            continue
        df = pd.read_csv(path)
        print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")

        print("\n-- columns / dtypes / non-null counts --")
        df.info()

        print("\n-- nulls per column --")
        nulls = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "nulls": df.isna().sum(),
            "pct_null": (df.isna().mean() * 100).round(1),
        })
        print(nulls.to_string())

        print(f"\n-- head ({N} rows) --")
        print(df.head(N).to_string(index=False))
        print()


if __name__ == "__main__":
    main()