"""
fill_missing.py

Fills missing (country, year) rows in team-features.csv.

Approach (per column, per team):
  - For each team we have rows for some subset of {2022, 2023, 2024, 2025, 2026}.
  - A "missing" year is one in that span that has no row.
  - We fill it by LINEAR INTERPOLATION between the team's nearest known years
    on either side. If a side is missing (the gap is at an edge of the team's
    series), we carry the nearest known value (forward/back fill).
  - All five numeric columns are filled this way. n_players is rounded to int.

Why interpolation over a flat 4-year mean:
  - These are per-team time series, so neighbours carry more signal than a
    global mean, and it reads more cleanly in the report.
  - Every gap in this dataset happens to be a 2022-2025 "pool" year (never the
    26-man 2026 squad), and every gap has at least one pool-year neighbour, so
    interpolation never has to reach into the structurally-different 2026 row.

Every filled cell is logged so it can be cited / audited.
"""

import pandas as pd
import numpy as np

SRC = "/home/claude/team-features.csv"
OUT = "/home/claude/team-features-filled.csv"
LOG = "/home/claude/fill_log.csv"

ALL_YEARS = [2022, 2023, 2024, 2025, 2026]
NUMERIC_COLS = ["n_players", "age_mean", "age_std", "value_mean", "value_std"]
INT_COLS = ["n_players"]  # round these after interpolation

df = pd.read_csv(SRC)

filled_rows = []   # new rows we synthesise
log_entries = []   # one entry per (country, year, column) filled

for country, grp in df.groupby("country", sort=True):
    grp = grp.set_index("year").sort_index()
    present_years = set(grp.index)
    missing_years = [y for y in ALL_YEARS if y not in present_years]
    if not missing_years:
        continue

    for y in missing_years:
        new_row = {"country": country, "year": y}
        for col in NUMERIC_COLS:
            series = grp[col].dropna()
            known_years = series.index.to_numpy()
            known_vals = series.to_numpy()

            # np.interp does linear interpolation between bracketing points and
            # clamps (carries the edge value) for years outside the known range.
            val = float(np.interp(y, known_years, known_vals))

            if col in INT_COLS:
                val = int(round(val))

            new_row[col] = val

            lo = known_years[known_years < y]
            hi = known_years[known_years > y]
            if len(lo) and len(hi):
                method = f"interpolated between {lo.max()} and {hi.min()}"
            elif len(lo):
                method = f"carried forward from {lo.max()}"
            else:
                method = f"carried back from {hi.min()}"

            log_entries.append({
                "country": country, "year": y, "column": col,
                "filled_value": new_row[col], "method": method,
            })

        filled_rows.append(new_row)

filled_df = pd.DataFrame(filled_rows)
out = (pd.concat([df, filled_df], ignore_index=True)
         .sort_values(["country", "year"])
         .reset_index(drop=True))

# keep original column order
out = out[["country", "year"] + NUMERIC_COLS]
out["n_players"] = out["n_players"].astype(int)

out.to_csv(OUT, index=False)
log_df = pd.DataFrame(log_entries)
log_df.to_csv(LOG, index=False)

print(f"Original rows : {len(df)}")
print(f"Rows added    : {len(filled_df)}")
print(f"Final rows    : {len(out)}")
print(f"Cells filled  : {len(log_df)}  ({len(filled_df)} rows x {len(NUMERIC_COLS)} cols)")
print()
print("Filled team-years:")
for _, r in filled_df.sort_values(["country", "year"]).iterrows():
    print(f"  {r['country']:25s} {int(r['year'])}")
print()
print("Per-cell log (value_mean shown as example column):")
vm = log_df[log_df["column"] == "value_mean"]
for _, r in vm.iterrows():
    print(f"  {r['country']:25s} {r['year']}  value_mean={r['filled_value']:>14,.0f}  ({r['method']})")