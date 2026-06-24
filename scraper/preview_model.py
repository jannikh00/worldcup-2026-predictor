"""
preview_model.py
================
Print the info and head of the model-ready match file.

Reads: ../data/matches_with_features_model.csv

Run:  python3 preview_model.py
"""

import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "../data")

FILE = os.path.join(DATA, "matches_with_features_model.csv")

# show every column, don't truncate the middle
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)
pd.set_option("display.max_colwidth", 18)


def main():
    df = pd.read_csv(FILE)

    print(FILE)
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")

    print("\n-- columns / dtypes / non-null counts --")
    df.info()

    print("\n-- head (10 rows) --")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()