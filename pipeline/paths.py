"""
paths.py
========
Single source of truth for every file the pipeline reads or writes.

Layout
------
  data/teams.csv        hand-curated seed: the 48 qualified nations + their
                        Transfermarkt slug/id. Committed.
  data/raw/             bulky or re-downloadable inputs. Git-ignored.
  data/processed/       pipeline outputs. Committed, so the repo is runnable
                        and every number in the README can be checked.

Paths are derived from this file's location, so scripts work from any cwd.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

# --- inputs ---------------------------------------------------------------
TEAMS_CSV = DATA / "teams.csv"              # country,slug,team_id
PAGES_DIR = RAW / "pages"                   # optional saved HTML: <country>_<year>.html

# --- raw / intermediate (git-ignored) -------------------------------------
RESULTS_CSV = RAW / "results.csv"           # cached copy of the martj42 results feed
PLAYERS_CSV = RAW / "players.csv"           # one row per (player, year)
PLAYERS_RESCRAPED_CSV = RAW / "players_rescraped.csv"

# --- processed outputs (committed) ----------------------------------------
TEAM_FEATURES_CSV = PROCESSED / "team_features.csv"          # one row per (country, year)
MATCHES_ELO_CSV = PROCESSED / "matches_with_elo.csv"         # match spine + pre-match Elo
MATCHES_FEATURES_CSV = PROCESSED / "matches_with_features.csv"  # + squad features (NaN where unmatched)
MATCHES_MODEL_CSV = PROCESSED / "matches_model.csv"          # model-ready subset


def ensure_dirs() -> None:
    """Create the data directories if they do not exist yet."""
    for d in (RAW, PROCESSED, PAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)
