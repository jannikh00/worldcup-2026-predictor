# World Cup 2026 — Match Result Prediction

An end-to-end pipeline that turns two public football data sources into a
model-ready table of international matches, and a four-protocol evaluation that
measures how much a common augment-before-you-split mistake distorts the result
— including a case where the *contaminated* protocol scores **lower**, for a
reason that isn't the obvious one.

The honest, deployment-shaped number: **a chronologically-split model reaches
~52% three-class accuracy against a 43% always-pick-home baseline, and almost
all of that signal comes from Elo alone.** The squad market-value and age
features I scraped add roughly 1.5 percentage points — inside the fold-to-fold
noise of the evaluation, so this repo reports them as *not demonstrably useful*
rather than as a win.

Built jointly as a shared project: the data pipeline and the evaluation layer
are mine, the classifier in `models/baseline.py` is a collaborator's. Full
breakdown under [Contributions](#contributions).

---

## Results

Three-class outcome (home win / away win / draw), 598 matches, 2022–2026.

| Model | Chronological split | Forward-chaining CV |
|---|---|---|
| Always predict home win | 0.427 | 0.417 ± 0.037 |
| Logistic regression, `elo_diff` only | 0.505 | 0.485 ± 0.043 |
| Logistic regression, all 5 features | **0.519** | **0.501 ± 0.053** |
| Random forest, all 5 features | 0.515 | 0.491 ± 0.064 |

*Chronological split* trains on everything before 2025-01-01 and tests on the
206 matches after it. *Forward-chaining CV* is five expanding-window folds.
Reproduce both with `python models/evaluate.py`.

Three things worth reading off that table:

1. **Elo does the work.** Going from the naive baseline to Elo-only is +7.8
   points. Going from Elo-only to all five features is +1.5, against a fold
   standard deviation of ±0.04–0.05. On this sample size that difference is not
   distinguishable from noise.
2. **Draws are effectively unpredictable.** Logistic regression never predicts
   a draw at all (recall 0.00); the random forest manages 0.15 recall. Draws are
   27.6% of matches, so this is where the accuracy ceiling lives.
3. **The single-split number is fragile.** The spread across CV folds is about
   the same size as the gap between the models being compared.

### On the evaluation protocol

The task is symmetric — swapping home and away should flip the label — so it is
tempting to augment the data by mirroring every match and negating the
difference features. That augmentation is sound, but applying it *before* the
train/test split is not: a match and its mirror are the same event, so they must
not land on opposite sides of the split.

`models/evaluate.py` runs four protocols side by side to make the cost of this
explicit:

| | Protocol | Accuracy (best model) |
|---|---|---|
| **A** | mirror → random split (contaminated) | 0.513 |
| **B** | random split, mirror train fold only | 0.567 |
| **C** | chronological split, mirror train fold only | 0.519 |
| **D** | forward-chaining CV | 0.501 |

Under protocol A, **194 of 217 test matches have their own mirrored twin in the
training set** (89%).

The interesting part is that A does *not* come out looking better. Mirroring the
test set as well removes home advantage — home and away wins become equally
frequent by construction — so the strongest single signal in the data is erased
from the thing being scored. Contamination pushes the number up, destroying the
home-win base rate pushes it down, and here the second effect wins.

The lesson is not "leakage inflates accuracy". It is that a broken protocol
produces a number that doesn't measure what its label claims, in whichever
direction. Protocol A isn't a pessimistic estimate of C; it's an estimate of a
different, artificially symmetric task. **C is the deployment-shaped number**,
because predicting a tournament means predicting forward in time.

---

## Pipeline

```
data/teams.csv ──► scrape_squads.py ──► players.csv ──► team_features.csv
                   (LLM extraction)                     (48 nations × 5 years)
                                                              │
martj42 results ──► build_elo.py ──► matches_with_elo.csv     │
   (49,477 matches                  (pre-match Elo, 2022+)    │
    since 1872)                              │                │
                                             ▼                ▼
                                      join_features.py ──► matches_with_features.csv
                                                                   │
                                                                   ▼
                                                          make_model_table.py
                                                                   │
                                                                   ▼
                                                          matches_model.csv (598 rows)
```

**`build_elo.py` — point-in-time Elo.** eloratings.net renders its tables with
JavaScript, so scraping it needs a headless browser. Instead this computes Elo
directly from raw results using the method eloratings documents: competition-
weighted K factors, a goal-difference multiplier, and +100 home advantage at
non-neutral venues. Ratings are warmed up over the **full history since 1872**
and only then filtered to 2022+. For every match it records each team's rating
*before* kickoff, then updates — so `elo_diff` never contains the result it is
used to predict.

**`scrape_squads.py` — LLM-based extraction.** Squad pages are read by an LLM
(`gpt-oss` via an OpenAI-compatible endpoint) that returns structured JSON,
rather than by CSS selectors that break whenever the page layout changes. The
division of labour is deliberate: **the model does extraction, plain Python does
arithmetic.** All aggregation (means, population standard deviations) happens in
`aggregate()`, because LLMs are unreliable at arithmetic and there is no reason
to trust one with a number you can compute exactly.

**`make_model_table.py` — leakage guard.** Drops `home_score`/`away_score`,
since the target is a deterministic function of them, and drops team names so
the model can't memorise which nations win. Keeps `date`, which a time-aware
split needs.

**`validate.py` — data quality gate.** 32 assertions over shape, completeness,
plausible ranges, derived-column consistency (`elo_diff` and the `*_diff`
features are recomputed and compared), and leakage. Exits non-zero, so it works
as a CI step.

---

## Quickstart

The processed data is committed, so the models run with no API key and no
scraping:

```bash
git clone https://github.com/jannikh00/worldcup-2026-predictor.git
cd worldcup-2026-predictor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python models/evaluate.py     # time-aware evaluation (the results above)
python pipeline/validate.py   # 32 data-quality checks
python models/baseline.py     # the original classifier
```

Rebuilding the derived data from the committed inputs:

```bash
python pipeline/build_elo.py        # needs network on first run
python pipeline/join_features.py
python pipeline/make_model_table.py
python pipeline/validate.py
```

Re-scraping squads from scratch additionally needs `NRP_API_KEY` (see
`.env.example`) and is the only step that requires credentials.

---

## Data provenance

| Source | What | Coverage | Notes |
|---|---|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | Every international result | 49,477 matches, 1872-11-30 → 2026-06-27 | Public dataset. Snapshot cached to `data/raw/results.csv` |
| [Transfermarkt](https://www.transfermarkt.us) | Squad age + market value | 48 nations × 5 seasons (2022–2026), 8,134 player-rows | Extracted via LLM; only aggregates are redistributed |
| `data/teams.csv` | The 48 qualified nations + URL slugs/ids | 48 rows | Hand-curated by me |

**The upstream results feed is append-only and updated after every
international window**, so `build_elo.py` caches its first download to
`data/raw/results.csv` and reuses it. Without that, re-running would silently
produce a different dataset than the one behind the numbers above. Delete the
cache to pull a fresh snapshot.

**Player-level Transfermarkt data is deliberately not committed.** `data/raw/`
is git-ignored; only the per-team aggregates (`team_features.csv`) are
redistributed. Scraping was low-volume, rate-limited, and for a non-commercial
portfolio project.

Everything under `data/processed/` **is** committed — about 1 MB — so that every
number in this README can be checked without re-running a scrape.

See [docs/data-dictionary.md](docs/data-dictionary.md) for column-level
definitions.

---

## Limitations

These are the things I'd want a reader to know before trusting any of it.

- **Only 598 of 4,608 matches are usable (13%).** Squad features exist only for
  the 48 nations qualified for 2026, and a match needs *both* sides covered. The
  rest of the join is retained in `matches_with_features.csv` with NaNs rather
  than silently dropped.
- **Squad features are joined by calendar year, which is not strictly
  point-in-time.** A match in March 2022 is matched to the 2022 squad snapshot,
  and a Transfermarkt season valuation reflects information from across that
  season. This is a mild look-ahead. Fixing it properly needs
  valuation-at-date, which Transfermarkt does not expose on the squad page.
- **The sample is friendly-heavy.** 224 of 598 matches (37%) are friendlies,
  where teams rotate squads and motivation varies, so squad-strength features
  are noisier than the aggregate suggests. 43% are at neutral venues.
- **Elo is reconstructed, not official.** The K-factor constants follow the
  published eloratings method but are lightly simplified, and every team starts
  at 1500 in 1872. Ratings converge long before 2022, but they are not
  identical to the official figures.
- **No hyperparameter tuning, and it would need a nested split.** The numbers
  above are untuned defaults. Tuning against the same test set the results are
  quoted from would reintroduce exactly the problem this repo is about.
- **336 of 8,134 player-rows have no market value** and are excluded from the
  value aggregates (ages are near-complete: 3 missing).
- **Accuracy is a weak metric for a three-class, imbalanced, low-signal task.**
  Log loss is reported alongside it; a calibration analysis would be a better
  next step than chasing accuracy.

---

## Repository layout

```
pipeline/              data engineering
  paths.py               single source of truth for all file locations
  scrape_squads.py       LLM extraction of squad age + market value
  rescrape_missing.py    targeted repair of failed (country, year) cells
  build_elo.py           point-in-time Elo from raw results
  join_features.py       attach squad features to each match
  make_model_table.py    row filter + leakage guard + identity drop
  validate.py            32-assertion data quality gate

models/
  baseline.py            original 3-class classifier
  evaluate.py            four-protocol time-aware evaluation

data/
  teams.csv              hand-curated seed (committed)
  raw/                   bulky / re-downloadable (git-ignored)
  processed/             pipeline outputs (committed)

docs/data-dictionary.md
```

---

## Contributions

Built jointly as a shared project.

**Mine — the data and evaluation layer:**

- Source selection and the decision to reconstruct Elo rather than scrape it
- `pipeline/` end to end: LLM-based squad extraction, the point-in-time Elo
  implementation, the feature join, the leakage guard, and the validation gate
- `models/evaluate.py`: the four-protocol time-aware evaluation, the
  contamination measurement, and the finding that the squad features are not
  demonstrably additive over Elo
- Reproducibility work: snapshot caching, the stable-sort fix described below,
  dependency pinning, and repository structure

**A collaborator's** — the classifier in `models/baseline.py`, kept as shipped
so the protocol comparison stays honest.

---

## Roadmap

- A second model, evaluated under the same four protocols, so the comparison is
  like-for-like rather than a fresh set of numbers
- Probability calibration (reliability curves, Brier score) — more informative
  than accuracy on a task with this much irreducible noise
- Ordered/ordinal treatment of the draw class instead of flat 3-class
- Bookmaker closing odds as a reference point, which is the real benchmark for
  match prediction
