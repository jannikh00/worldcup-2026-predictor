"""
elo.py
============
build the match "spine" and attach point-in-time Elo ratings.

WHY WE DON'T SCRAPE eloratings.net DIRECTLY
-------------------------------------------
eloratings.net renders its tables with JavaScript, so a plain
requests + BeautifulSoup scrape returns an empty page (you'd need a full
headless browser like Playwright, which is slow and fragile). Instead we:

  1. PULL a clean feed of every international match result
     (martj42's public dataset, hosted as a raw CSV on GitHub), and
  2. COMPUTE our OWN point-in-time Elo from those results, using the method
     eloratings.net documents (match-importance weighting, home advantage,
     and a goal-difference adjustment).

This gives us, for every match, each team's Elo BEFORE kickoff -> the
`elo_diff` feature, with no data leakage. Computing Elo ourselves is also a
genuine original contribution and removes any dependency on a fragile website.

DATE CUTOFF
-----------
Elo needs history to "warm up", so we compute over ALL matches since 1872,
then keep only matches from START_YEAR (2022) onward in the final output, so
the 2022 World Cup is included.
"""

import io
import pandas as pd
import requests
from pathlib import Path

# ------------------------------- Config -----------------------------------
START_YEAR = 2022          # keep matches from this year onward (2022 -> last World Cup included)
INITIAL_ELO = 1500.0       # every team starts here; ratings converge long before 2022
HOME_ADVANTAGE = 100.0     # Elo points added to the home team (eloratings uses ~100); 0 at neutral venues
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_CSV = DATA_DIR / "matches_with_elo.csv"

# martj42 international results (matches + scores). Branch is usually "master";
# we also try "main" as a fallback in case the repo was renamed.
RESULTS_URLS = [
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "https://raw.githubusercontent.com/martj42/international_results/main/results.csv",
]


# --------------------------- Elo helper functions -------------------------
def expected_score(elo_team, elo_opponent, home_adv=0.0):
    """Expected result for `team` vs `opponent`, on a 0..1 scale (1 = win, 0.5 = draw,
    0 = loss), per the standard Elo formula. `home_adv` is added to the team's
    effective rating (use +HOME_ADVANTAGE for the home side, 0 at a neutral venue)."""
    diff = (elo_team + home_adv) - elo_opponent
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def k_factor(tournament, goal_diff):
    """eloratings-style match weight K.

    Base value reflects how important the competition is; it is then scaled up
    for bigger winning margins (a 4-0 win moves ratings more than a 1-0 win)."""
    t = (tournament or "").lower()

    # Base importance (eloratings constants, lightly simplified)
    if "world cup" in t and "qual" not in t:
        base = 60                      # World Cup finals
    elif any(c in t for c in ["euro", "copa", "african cup of nations",
                              "afc asian", "gold cup", "confederations"]) and "qual" not in t:
        base = 50                      # continental championships
    elif "qualif" in t or "nations league" in t:
        base = 40                      # qualifiers / Nations League
    elif "friendly" in t:
        base = 20                      # friendlies
    else:
        base = 30                      # anything else (minor cups, etc.)

    # Goal-difference multiplier (eloratings rule)
    gd = abs(goal_diff)
    if gd <= 1:
        mult = 1.0
    elif gd == 2:
        mult = 1.5
    elif gd == 3:
        mult = 1.75
    else:                              # 4 or more
        mult = 1.75 + (gd - 3) / 8.0
    return base * mult


def match_result(home_score, away_score):
    """Return (home_points, away_points, label) where points are 1 / 0.5 / 0
    and label is 'H' (home win), 'A' (away win), or 'D' (draw)."""
    if home_score > away_score:
        return 1.0, 0.0, "H"
    elif home_score < away_score:
        return 0.0, 1.0, "A"
    else:
        return 0.5, 0.5, "D"


# ----------------------------- Core computation ---------------------------
def compute_elo(df):
    """Walk matches in date order. For each match, record both teams' PRE-match
    Elo (what the model is allowed to see), then update their ratings from the
    result. Returns (output_dataframe, final_ratings_dict)."""
    ratings = {}          # team name -> current Elo
    rows = []             # collected output rows

    for i, m in enumerate(df.itertuples(index=False)):
        home, away = m.home_team, m.away_team

        # current ratings (a team we've never seen starts at INITIAL_ELO)
        rh = ratings.get(home, INITIAL_ELO)
        ra = ratings.get(away, INITIAL_ELO)

        # home advantage applies only at non-neutral venues
        adv = 0.0 if bool(m.neutral) else HOME_ADVANTAGE

        # expected vs actual
        exp_h = expected_score(rh, ra, home_adv=adv)
        exp_a = 1.0 - exp_h
        pts_h, pts_a, label = match_result(m.home_score, m.away_score)

        # match weight
        k = k_factor(m.tournament, m.home_score - m.away_score)

        # record the PRE-match state -> this is leakage-free and is what we train on
        rows.append({
            "date": m.date,
            "home_team": home,
            "away_team": away,
            "home_score": m.home_score,
            "away_score": m.away_score,
            "tournament": m.tournament,
            "neutral": bool(m.neutral),
            "elo_home_pre": round(rh, 1),
            "elo_away_pre": round(ra, 1),
            "elo_diff": round((rh + adv) - ra, 1),   # home-perspective pre-match Elo gap
            "result": label,
        })

        # update ratings AFTER recording the pre-match snapshot
        ratings[home] = rh + k * (pts_h - exp_h)
        ratings[away] = ra + k * (pts_a - exp_a)

        # progress print so you can see it's working on the full ~48k-match history
        if (i + 1) % 5000 == 0:
            print(f"  ...processed {i + 1:,} matches")

    return pd.DataFrame(rows), ratings


def load_results():
    """Download the international results CSV from GitHub (tries both branch names)."""
    for url in RESULTS_URLS:
        try:
            print(f"Downloading results from: {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            print(f"  OK - {len(df):,} rows loaded")
            return df
        except Exception as e:
            print(f"  failed ({e}); trying next URL...")
    raise RuntimeError("Could not download results.csv from any known URL.")


def main():
    print("Script started ...\n")
    # 1) load raw results
    df = load_results()
    print(f"Columns found: {list(df.columns)}")

    # 2) basic cleaning
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    # IMPORTANT: parse `neutral` properly. read_csv may load it as the strings
    # "True"/"False"; bool("False") is True, so coerce explicitly.
    df["neutral"] = (
        df["neutral"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    )
    df = df.sort_values("date").reset_index(drop=True)
    print(f"After cleaning: {len(df):,} matches, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    # 3) compute Elo over the FULL history (so ratings are warmed up before 2022)
    print("Computing point-in-time Elo over full history...")
    out, final_ratings = compute_elo(df)

    # 4) sanity check: strongest teams right now should be elite nations
    top = sorted(final_ratings.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop 10 teams by current Elo (sanity check - expect elite nations):")
    for team, rating in top:
        print(f"  {team:<22} {rating:7.1f}")

    # 5) keep only recent matches for the final dataset
    out = out[out["date"].dt.year >= START_YEAR].reset_index(drop=True)
    print(f"\nKept {len(out):,} matches from {START_YEAR} onward.")

    # 6) show a few example rows so you can eyeball the output
    print("\nSample of the output:")
    print(out.head(8).to_string(index=False))

    # 7) save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()