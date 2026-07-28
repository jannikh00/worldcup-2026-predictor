# Data dictionary

Column-level reference for every file under `data/`. Files marked *committed*
are in git; files marked *git-ignored* are regenerable and excluded to keep the
clone small and to avoid redistributing player-level scraped data.

---

## `data/teams.csv` — committed

Hand-curated seed listing the 48 nations qualified for the 2026 World Cup.
This is the only file in the repository that was assembled by hand.

| Column | Type | Description |
|---|---|---|
| `country` | str | Nation name. **This is the join key** against `home_team` / `away_team` in the results feed, so the spelling must match that feed exactly. |
| `slug` | str | Transfermarkt URL slug, e.g. `brasilien`. |
| `team_id` | int | Transfermarkt `verein` id, e.g. `3439`. |

URLs are built as
`https://www.transfermarkt.us/<slug>/kader/verein/<team_id>/saison_id/<year>`.

---

## `data/raw/results.csv` — git-ignored

Cached snapshot of the [martj42/international_results](https://github.com/martj42/international_results)
feed: 49,477 matches, 1872-11-30 → 2026-06-27. Cached on first run because the
feed is append-only and updated after every international window; without the
cache, re-running the pipeline would silently produce a different dataset.

Columns used: `date`, `home_team`, `away_team`, `home_score`, `away_score`,
`tournament`, `neutral`.

---

## `data/raw/players.csv` — git-ignored

One row per (player, season). 8,134 rows. Not redistributed — only the
per-team aggregates derived from it are committed.

| Column | Type | Description |
|---|---|---|
| `name` | str | Player name as listed. |
| `age` | int | Age in years at the time of the snapshot. 3 nulls. |
| `market_value_eur` | int | Market value in whole euros; `null` where Transfermarkt shows no value. 336 nulls. |
| `country` | str | Nation, matching `teams.csv`. |
| `year` | int | Transfermarkt `saison_id`, 2022–2026. |

`data/raw/players_rescraped.csv` has the same schema and holds only the rows
recovered by `rescrape_missing.py`.

---

## `data/processed/team_features.csv` — committed

One row per (country, year). 240 rows = 48 nations × 5 seasons, complete with
no gaps (`validate.py` asserts this).

| Column | Type | Description |
|---|---|---|
| `country` | str | Nation. |
| `year` | int | Season, 2022–2026. |
| `n_players` | int | Squad size the aggregates were computed from. Range 23–44, median 37. |
| `age_mean` | float | Mean squad age. Observed range 23.7–30.1. |
| `age_std` | float | **Population** standard deviation of age (`ddof=0`). |
| `value_mean` | float | Mean market value in euros, over players with a listed value. |
| `value_std` | float | Population standard deviation of market value. |

Population rather than sample standard deviation is intentional: a squad is the
entire group of interest, not a sample drawn from a larger population.

---

## `data/processed/matches_with_elo.csv` — committed

The match spine with point-in-time Elo. 4,608 matches, 2022-01-01 → 2026-06-21.
Elo is warmed up over the full history since 1872, then filtered to 2022+.

| Column | Type | Description |
|---|---|---|
| `date` | date | Match date. |
| `home_team`, `away_team` | str | Team names as in the source feed. |
| `home_score`, `away_score` | int | Full-time goals. |
| `tournament` | str | Competition name; drives the Elo K factor. |
| `neutral` | bool | True at a neutral venue. Home advantage is applied only when False. |
| `elo_home_pre`, `elo_away_pre` | float | Each team's rating **before** kickoff. |
| `elo_diff` | float | `(elo_home_pre + advantage) - elo_away_pre`, where advantage is 100.0 off-neutral and 0.0 at neutral venues. |
| `result` | str | `H` home win, `A` away win, `D` draw. |

`elo_*_pre` are recorded before the ratings are updated with the match result,
so they contain no information about the match being predicted.

---

## `data/processed/matches_with_features.csv` — committed

`matches_with_elo.csv` plus squad features for both sides, joined on
`(team, calendar year of the match)`. Same 4,608 rows — matches involving a
nation outside the 48 keep their row and carry `NaN` features, rather than
being silently dropped.

Adds, for each of `age_mean`, `age_std`, `value_mean`, `value_std`:

| Column | Description |
|---|---|
| `home_<feature>` | Home side's value. |
| `away_<feature>` | Away side's value. |
| `<feature>_diff` | `home_<feature> - away_<feature>`. |
| `year` | Calendar year of the match; the join key. |

Coverage: both sides matched for 598 rows; home only 887; away only 616;
neither 2,507.

---

## `data/processed/matches_model.csv` — committed

The model-ready table. 598 rows × 19 columns, sorted chronologically with a
stable sort, 2022-01-05 → 2026-06-21.

Derived from `matches_with_features.csv` by:

1. keeping only rows where **both** teams have squad features;
2. dropping `home_score` and `away_score` — the target `result` is a
   deterministic function of them, so shipping them in a table labelled
   "model-ready" would be a target-leakage trap;
3. dropping `home_team`, `away_team`, `year` so the model cannot key off team
   identity.

`date` is deliberately retained: the time-aware split in `models/evaluate.py`
needs it.

**Class balance:** `H` 43.8%, `A` 28.6%, `D` 27.6%.
**Competition mix:** 224 friendlies, 86 World Cup qualifiers, 86 World Cup
finals, 52 Nations League, remainder continental tournaments. 42.8% neutral.

### Features used by the models

The five symmetric difference features. They are symmetric by construction:
swapping home and away negates each one, which is what makes the mirroring
augmentation in `models/evaluate.py` valid.

| Feature | Description |
|---|---|
| `elo_diff` | Pre-match Elo gap, home perspective, including home advantage. |
| `age_mean_diff` | Difference in mean squad age. |
| `age_std_diff` | Difference in squad age spread. |
| `value_mean_diff` | Difference in mean market value (euros). |
| `value_std_diff` | Difference in market value spread. |

**Target:** `result` ∈ {`H`, `A`, `D`}, encoded `H`→1, `A`→0, `D`→2.

The remaining columns (`elo_home_pre`, `elo_away_pre`, the per-side
`home_*`/`away_*` levels, `tournament`, `neutral`) are carried for analysis and
are available as candidate features; the models in `models/` do not currently
use them.
