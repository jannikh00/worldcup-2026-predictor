"""
validate.py
===========
Data-quality gate for everything under data/. Run it after the pipeline; it
exits non-zero if any check fails, so it works as a CI step.

The checks are grouped by what they protect against:

  SHAPE      - the files exist and have the columns downstream code expects
  COMPLETE   - the 48 x 5 squad grid is fully populated, no silent NaNs
  PLAUSIBLE  - values sit in physically sensible ranges (ages, squad sizes)
  CONSISTENT - derived columns really are derived from their inputs
               (elo_diff, the *_diff features, and `result` vs the scores)
  LEAKAGE    - the model table does not carry the answer or team identity

The CONSISTENT and LEAKAGE groups are the ones that matter. A join that
silently misaligns, or a score column surviving into the model table, is the
kind of bug that shows up as a suspiciously good accuracy rather than as a
crash.

Run:  python pipeline/validate.py
"""

import sys

import pandas as pd

from paths import (
    TEAMS_CSV, TEAM_FEATURES_CSV, MATCHES_ELO_CSV,
    MATCHES_FEATURES_CSV, MATCHES_MODEL_CSV,
)

HOME_ADVANTAGE = 100.0        # must match build_elo.py
YEARS = [2022, 2023, 2024, 2025, 2026]
N_TEAMS = 48

BASE_FEATURES = ["age_mean", "age_std", "value_mean", "value_std"]

_results = []


def check(group, name, ok, detail=""):
    _results.append((group, name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main():
    print("=" * 74)
    print("DATA VALIDATION")
    print("=" * 74)

    # ---------------------------------------------------------------- SHAPE
    print("\nSHAPE")
    for path in (TEAMS_CSV, TEAM_FEATURES_CSV, MATCHES_ELO_CSV,
                 MATCHES_FEATURES_CSV, MATCHES_MODEL_CSV):
        check("SHAPE", f"exists: {path.name}", path.exists(), str(path) if not path.exists() else "")
    if not all(r[2] for r in _results):
        print("\nMissing files - run the pipeline first. Aborting.")
        return 1

    teams = pd.read_csv(TEAMS_CSV)
    feat = pd.read_csv(TEAM_FEATURES_CSV)
    elo = pd.read_csv(MATCHES_ELO_CSV, parse_dates=["date"])
    joined = pd.read_csv(MATCHES_FEATURES_CSV, parse_dates=["date"])
    model = pd.read_csv(MATCHES_MODEL_CSV, parse_dates=["date"])

    check("SHAPE", "teams.csv columns",
          set(teams.columns) == {"country", "slug", "team_id"}, str(list(teams.columns)))
    check("SHAPE", "team_features.csv columns",
          set(BASE_FEATURES) <= set(feat.columns), str(list(feat.columns)))

    # ------------------------------------------------------------- COMPLETE
    print("\nCOMPLETE")
    check("COMPLETE", f"teams.csv has {N_TEAMS} nations", len(teams) == N_TEAMS, f"n={len(teams)}")
    check("COMPLETE", "teams.csv country unique", teams["country"].is_unique)
    check("COMPLETE", "teams.csv team_id unique", teams["team_id"].is_unique)

    expected_cells = N_TEAMS * len(YEARS)
    check("COMPLETE", f"squad grid is {N_TEAMS} x {len(YEARS)} = {expected_cells}",
          len(feat) == expected_cells, f"n={len(feat)}")
    check("COMPLETE", "squad grid has no gaps",
          feat.groupby("year").size().eq(N_TEAMS).all(),
          str(dict(feat.groupby("year").size())))
    check("COMPLETE", "no nulls in team features",
          feat[BASE_FEATURES].notna().all().all(),
          str(dict(feat[BASE_FEATURES].isna().sum())))
    check("COMPLETE", "feature countries match teams.csv",
          set(feat["country"]) == set(teams["country"]),
          str(sorted(set(feat["country"]) ^ set(teams["country"]))))
    check("COMPLETE", "no nulls in model table",
          model.notna().all().all(), str(dict(model.isna().sum()[lambda s: s > 0])))

    # ------------------------------------------------------------ PLAUSIBLE
    print("\nPLAUSIBLE")
    check("PLAUSIBLE", "mean squad age in 18..40",
          feat["age_mean"].between(18, 40).all(),
          f"min={feat['age_mean'].min()} max={feat['age_mean'].max()}")
    check("PLAUSIBLE", "squad size in 11..60",
          feat["n_players"].between(11, 60).all(),
          f"min={feat['n_players'].min()} max={feat['n_players'].max()}")
    check("PLAUSIBLE", "market values non-negative", (feat["value_mean"] >= 0).all())
    check("PLAUSIBLE", "std columns non-negative",
          (feat["age_std"] >= 0).all() and (feat["value_std"] >= 0).all())
    check("PLAUSIBLE", "no duplicate (country, year)",
          not feat.duplicated(["country", "year"]).any())

    # ----------------------------------------------------------- CONSISTENT
    print("\nCONSISTENT")
    # `result` must agree with the scores it was derived from
    derived = pd.Series("D", index=elo.index)
    derived[elo["home_score"] > elo["away_score"]] = "H"
    derived[elo["home_score"] < elo["away_score"]] = "A"
    check("CONSISTENT", "result agrees with scores",
          derived.eq(elo["result"]).all(),
          f"{(~derived.eq(elo['result'])).sum()} mismatches")

    # elo_diff must equal (home + advantage) - away, advantage only off-neutral
    adv = (~elo["neutral"].astype(bool)) * HOME_ADVANTAGE
    recomputed = (elo["elo_home_pre"] + adv - elo["elo_away_pre"]).round(1)
    check("CONSISTENT", "elo_diff = (home_pre + adv) - away_pre",
          (recomputed - elo["elo_diff"]).abs().le(0.11).all(),
          f"max abs err={(recomputed - elo['elo_diff']).abs().max():.3f}")

    # the *_diff features must equal home minus away
    diff_ok = True
    worst = 0.0
    for c in BASE_FEATURES:
        err = (joined[f"home_{c}"] - joined[f"away_{c}"] - joined[f"{c}_diff"]).abs().max()
        worst = max(worst, 0.0 if pd.isna(err) else err)
        diff_ok &= bool(pd.isna(err) or err < 1e-6)
    check("CONSISTENT", "feature diffs = home - away", diff_ok, f"max abs err={worst:.2e}")

    check("CONSISTENT", "match dates are chronological in model table",
          model["date"].is_monotonic_increasing)
    check("CONSISTENT", "model rows = fully-featured joined rows",
          len(model) == (joined["home_age_mean"].notna() & joined["away_age_mean"].notna()).sum(),
          f"model={len(model)}")

    # ------------------------------------------------------------- LEAKAGE
    print("\nLEAKAGE")
    for col in ("home_score", "away_score"):
        check("LEAKAGE", f"model table excludes {col}", col not in model.columns)
    for col in ("home_team", "away_team", "year"):
        check("LEAKAGE", f"model table excludes {col}", col not in model.columns)
    check("LEAKAGE", "model table keeps date (needed for time-aware split)",
          "date" in model.columns)

    # Elo is a pre-match snapshot, so it must not encode this match's outcome.
    # If it did, elo_diff alone would separate the classes almost perfectly.
    sep = model.groupby("result")["elo_diff"].mean()
    overlap = model["elo_diff"].between(sep.min(), sep.max()).mean()
    check("LEAKAGE", "elo_diff does not trivially separate the target",
          overlap > 0.20, f"{overlap:.1%} of rows lie between class means")

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 74)
    failed = [r for r in _results if not r[2]]
    print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("\nFAILED:")
        for group, name, _, detail in failed:
            print(f"  {group}: {name}  {detail}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
