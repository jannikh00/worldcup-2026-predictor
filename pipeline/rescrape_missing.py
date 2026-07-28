"""
rescrape_missing.py
===================
Targeted repair: re-scrape ONLY the 8 (country, year) cells that failed in the
full run, then merge them into team_features.csv.

Why a separate script instead of re-running the whole grid:
  - It hits 8 pages instead of 48*5 = 240, so it's fast and polite.
  - It REUSES scrape_transfermarkt's own functions (build_url, get_html,
    extract_players, aggregate), so the 8 repaired rows are computed with the
    exact same extraction + population-std logic as the other 232 rows. No
    methodological drift between the original cells and the repaired ones.
"""

import os
import csv

import pandas as pd

# Reuse the real pipeline so aggregation stays identical to the 232 good rows.
from transfermarkt import (
    build_url, get_html, extract_players, aggregate,
    TEAMS_CSV, FEATURES_CSV,
)

# The exact cells that failed in the full scrape (all are 2022-2025 pool years).
MISSING = [
    ("Belgium",        2025),
    ("Cape Verde",     2024),
    ("Czech Republic", 2022),
    ("Egypt",          2024),
    ("Spain",          2024),
    ("Sweden",         2025),
    ("Tunisia",        2023),
    ("Turkey",         2024),
]

OUT_FEATURES = os.path.join(os.path.dirname(FEATURES_CSV), "team_features_rescraped.csv")
OUT_PLAYERS  = os.path.join(os.path.dirname(FEATURES_CSV), "players_rescraped.csv")


def load_team_lookup():
    """country -> (slug, team_id), from the same teams.csv the main script uses."""
    lut = {}
    with open(TEAMS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lut[row["country"].strip()] = (row["slug"].strip(), row["team_id"].strip())
    return lut


def main():
    lut = load_team_lookup()

    all_players = []
    failures = []
    for country, year in MISSING:
        if country not in lut:
            print(f"  !! {country!r} not in {TEAMS_CSV} - check the spelling there")
            failures.append((country, year, "not in teams.csv"))
            continue
        slug, team_id = lut[country]
        url = build_url(slug, team_id, year)
        print(f"  {country} {year}")
        try:
            html = get_html(country, year, url)
            players = extract_players(html)
            if not players:
                raise ValueError("0 players extracted")
            for p in players:
                p["country"], p["year"] = country, year
            all_players.extend(players)
            print(f"    extracted {len(players)} players")
        except Exception as e:
            print(f"    FAILED ({e})")
            failures.append((country, year, str(e)))

    if not all_players:
        raise SystemExit(
            "Nothing extracted. Most likely the 8 HTML pages aren't saved under "
            "data/pages/ yet (Transfermarkt 403s direct requests)."
        )

    players_df = pd.DataFrame(all_players)
    players_df.to_csv(OUT_PLAYERS, index=False)
    print(f"\nSaved {len(players_df)} player rows -> {OUT_PLAYERS}")

    # Aggregate the new cells with the SAME function as the main pipeline.
    new_rows = aggregate(players_df)
    print(f"Aggregated {len(new_rows)} (country, year) rows:")
    print(new_rows.to_string(index=False))

    # Merge: drop any existing copy of these cells, append the fresh ones.
    base = pd.read_csv(FEATURES_CSV)
    key = ["country", "year"]
    got = set(map(tuple, new_rows[key].values))
    base = base[~base.set_index(key).index.isin(got)].reset_index(drop=True)
    merged = (pd.concat([base, new_rows], ignore_index=True)
                .sort_values(key)
                .reset_index(drop=True))
    merged.to_csv(OUT_FEATURES, index=False)

    print(f"\nMerged file -> {OUT_FEATURES}  ({len(merged)} rows)")
    # print("Review it, then replace data/team_features.csv with it if it looks right.")

    still_missing = [m for m in MISSING if m not in got]
    if still_missing:
        print("\nStill missing (save their HTML pages and re-run):")
        for c, y in still_missing:
            print(f"  {c} {y}")


if __name__ == "__main__":
    main()