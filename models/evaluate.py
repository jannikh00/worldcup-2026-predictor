"""
evaluate.py
===========
A leakage-resistant, time-aware evaluation of the match-result task, and a
direct measurement of how much the baseline's protocol inflates its score.

WHY THE BASELINE'S NUMBER IS OPTIMISTIC
---------------------------------------
baseline.py does two things that leak:

  1. MIRROR-THEN-SPLIT. Every match is duplicated with home/away swapped, the
     five difference features negated and the label flipped. That augmentation
     is defensible in itself - the task is symmetric, so the model should be
     too - but it is applied BEFORE the split. A match and its own mirror are
     the same event, and they can land on opposite sides of the train/test
     line. The model then "predicts" a fixture it has already memorised, with
     the sign flipped.

  2. RANDOM SPLIT ON TEMPORAL DATA. Elo ratings are a running function of
     earlier results, so a match from 2026 in the training set carries
     information about 2022 fixtures in the test set. The deployment task is
     strictly forward-looking: predict a tournament that has not happened yet.
     A shuffled split does not measure that.

WHAT THIS SCRIPT DOES INSTEAD
-----------------------------
  * PROTOCOL A - reproduces the baseline's mirror-then-random-split number.
  * PROTOCOL B - keeps the random split but mirrors INSIDE the training fold
    only, so no test fixture has its twin in training. Isolates the cost of
    leak (1).
  * PROTOCOL C - chronological split: train on everything before a cutoff
    date, test on everything after. Mirroring again confined to training.
    This is the honest number, and it is the one the README quotes.
  * PROTOCOL D - forward-chaining CV: expanding-window folds over time, which
    is the time-series analogue of k-fold and gives a spread rather than one
    fragile point estimate.

Every protocol is scored against two reference points, because 3-class
accuracy on its own is easy to misread:
  * MAJORITY - always predict the most common training class.
  * ELO-ONLY - logistic regression on the single elo_diff feature. If the
    squad-value features add nothing over Elo, this is where it shows.

Run:  python models/evaluate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "matches_model.csv"

DIFF_COLS = ["elo_diff", "age_mean_diff", "age_std_diff", "value_mean_diff", "value_std_diff"]
LABEL_MAP = {"A": 0, "H": 1, "D": 2}          # away win / home win / draw
TARGET_NAMES = ["Away win", "Home win", "Draw"]
CLASSES = [0, 1, 2]

SEED = 42
TEST_FRACTION = 0.2       # used by the random-split protocols
N_FOLDS = 5               # forward-chaining folds


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df["label"] = df["result"].map(LABEL_MAP)
    df = df.sort_values("date").reset_index(drop=True)
    # match_id identifies the real-world event, so a match and its mirror share
    # one id. That is what lets us keep them on the same side of a split.
    df["match_id"] = np.arange(len(df))
    return df


def mirror(df):
    """Return df plus its home/away-swapped twin.

    The task is symmetric: swapping the two teams should flip a home win to an
    away win and leave a draw a draw. Negating the difference features encodes
    that. match_id is carried over unchanged so the twin stays traceable.
    """
    flipped = df.copy()
    flipped[DIFF_COLS] = -flipped[DIFF_COLS]
    flipped["label"] = flipped["label"].map({0: 1, 1: 0, 2: 2})
    return pd.concat([df, flipped], ignore_index=True)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def make_models():
    """Fresh, unfitted estimators. Scaling lives inside the pipeline so it is
    fitted on training data only - fitting a scaler on the full dataset is
    itself a (mild) leak.

    "Always home win" is the naive reference, and it is deliberately a CONSTANT
    predictor rather than most_frequent. Mirroring the training fold makes home
    and away wins exactly equally frequent by construction, so most_frequent
    would break the tie arbitrarily (it picks away) and report a meaningless
    number. Home win is the majority class of the real, unmirrored data and is
    the standard naive baseline in football modelling.
    """
    return {
        "Always home win": DummyClassifier(strategy="constant", constant=LABEL_MAP["H"]),
        "Elo only (LogReg)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "RandomForest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=200, random_state=SEED, min_samples_leaf=5),
        ),
    }


def features_for(name):
    return ["elo_diff"] if name == "Elo only (LogReg)" else DIFF_COLS


def score(model, train, test, name):
    cols = features_for(name)
    model.fit(train[cols], train["label"])
    pred = model.predict(test[cols])
    prob = model.predict_proba(test[cols])
    return {
        "accuracy": accuracy_score(test["label"], pred),
        # labels= pins column order so log_loss is correct even if a fold is
        # missing a class entirely
        "log_loss": log_loss(test["label"], prob, labels=CLASSES),
    }


def run_protocol(train, test, header, note):
    print("\n" + "=" * 74)
    print(header)
    print("-" * 74)
    print(note)
    print(f"train: {len(train):5d} rows   test: {len(test):5d} rows")
    print(f"test class balance: " +
          "  ".join(f"{k}={v:.1%}" for k, v in
                    test['label'].map({v: k for k, v in LABEL_MAP.items()})
                    .value_counts(normalize=True).items()))
    print(f"{'model':<22}{'accuracy':>12}{'log loss':>12}")
    results = {}
    for name, model in make_models().items():
        r = score(model, train, test, name)
        results[name] = r
        print(f"{name:<22}{r['accuracy']:>12.4f}{r['log_loss']:>12.4f}")
    return results


# --------------------------------------------------------------------------
# protocols
# --------------------------------------------------------------------------
def protocol_a(df):
    """Baseline's own protocol: mirror everything, then split at random."""
    full = mirror(df)
    train, test = train_test_split(
        full, test_size=TEST_FRACTION, random_state=SEED, stratify=full["label"]
    )
    shared = len(set(train["match_id"]) & set(test["match_id"]))
    note = (f"Mirror -> random split. {shared} of {test['match_id'].nunique()} test matches\n"
            f"have their twin in the training set ({100*shared/test['match_id'].nunique():.0f}% leaked).")
    return run_protocol(train, test, "PROTOCOL A - baseline protocol (leaky)", note), shared


def protocol_b(df):
    """Random split on real matches; mirror only the training fold."""
    train_ids, test_ids = train_test_split(
        df["match_id"], test_size=TEST_FRACTION, random_state=SEED, stratify=df["label"]
    )
    train = mirror(df[df["match_id"].isin(train_ids)])
    test = df[df["match_id"].isin(test_ids)]
    note = "Split real matches first, then mirror TRAIN only. No twin leakage;\nstill ignores time."
    return run_protocol(train, test, "PROTOCOL B - random split, no twin leakage", note)


def protocol_c(df, cutoff):
    """Chronological split - the honest, deployment-shaped evaluation."""
    train = mirror(df[df["date"] < cutoff])
    test = df[df["date"] >= cutoff]
    note = (f"Train on matches before {cutoff.date()}, test on matches after.\n"
            f"Mirror TRAIN only. This is the number the README quotes.")
    return run_protocol(train, test, "PROTOCOL C - chronological split (honest)", note)


def protocol_d(df):
    """Forward-chaining CV: expanding window, each fold tests the next block."""
    print("\n" + "=" * 74)
    print("PROTOCOL D - forward-chaining CV (expanding window)")
    print("-" * 74)
    print(f"{N_FOLDS} folds; each trains on all matches before the fold and tests on it.")

    bounds = np.linspace(0, len(df), N_FOLDS + 2, dtype=int)
    per_model = {name: [] for name in make_models()}

    for i in range(1, N_FOLDS + 1):
        tr_end, te_end = bounds[i], bounds[i + 1]
        train = mirror(df.iloc[:tr_end])
        test = df.iloc[tr_end:te_end]
        if test.empty or train["label"].nunique() < 3:
            continue
        for name, model in make_models().items():
            per_model[name].append(score(model, train, test, name)["accuracy"])
        print(f"  fold {i}: train n={tr_end:4d} (to {df.iloc[tr_end-1]['date'].date()}) "
              f" test n={len(test):3d}")

    print(f"\n{'model':<22}{'mean acc':>12}{'std':>10}{'folds':>8}")
    summary = {}
    for name, accs in per_model.items():
        if not accs:
            continue
        summary[name] = (float(np.mean(accs)), float(np.std(accs)))
        print(f"{name:<22}{np.mean(accs):>12.4f}{np.std(accs):>10.4f}{len(accs):>8d}")
    return summary


# --------------------------------------------------------------------------
def main():
    df = load()
    cutoff = pd.Timestamp("2025-01-01")

    print("=" * 74)
    print("MATCH-RESULT EVALUATION")
    print("=" * 74)
    print(f"rows: {len(df)}   {df['date'].min().date()} -> {df['date'].max().date()}")
    dist = df["result"].value_counts(normalize=True)
    print("class balance: " + "  ".join(f"{k}={v:.1%}" for k, v in dist.items()))
    print(f"features: {DIFF_COLS}")

    a, leaked = protocol_a(df)
    b = protocol_b(df)
    c = protocol_c(df, cutoff)
    d = protocol_d(df)

    # ---- the headline comparison ----
    print("\n" + "=" * 74)
    print("SUMMARY - accuracy by protocol")
    print("=" * 74)
    print(f"{'model':<22}{'A leaky':>11}{'B no-twin':>11}{'C time':>11}{'D fwd-CV':>11}")
    for name in make_models():
        row = f"{name:<22}{a[name]['accuracy']:>11.4f}{b[name]['accuracy']:>11.4f}{c[name]['accuracy']:>11.4f}"
        row += f"{d[name][0]:>11.4f}" if name in d else f"{'-':>11}"
        print(row)

    # ---- what the comparison actually shows ----
    print("\n" + "=" * 74)
    print("READING THE TABLE")
    print("=" * 74)
    print(f"""
Protocol A is contaminated: {leaked} of its test matches have their mirrored
twin in the training set. That is a genuine defect - the test set is not
independent of the training set, so A does not measure generalisation.

But A does NOT come out looking better, and it is worth being precise about
why. Mirroring the TEST set as well as the training set removes the home
advantage: home and away wins become equally frequent by construction, so the
single strongest signal in the data is erased from the thing being scored.
Two effects run in opposite directions - contamination helps, destroying the
home-win base rate hurts - and here the second dominates.

The lesson is not "leakage inflates accuracy". It is that a broken protocol
produces a number that does not mean what its label says, in whichever
direction. A is not a pessimistic estimate of B or C; it is an estimate of a
different, artificially symmetric task.

Protocol C is the deployment-shaped number. Protocol D shows how fragile any
single split is here: the fold-to-fold spread is roughly the same size as the
gap between the models.
""".strip())

    # ---- detail on the honest protocol ----
    print("\n" + "=" * 74)
    print("PROTOCOL C detail - per-class report")
    print("=" * 74)
    train = mirror(df[df["date"] < cutoff])
    test = df[df["date"] >= cutoff]
    for name in ["LogReg", "RandomForest"]:
        model = make_models()[name]
        cols = features_for(name)
        model.fit(train[cols], train["label"])
        pred = model.predict(test[cols])
        print(f"\n--- {name} ---")
        print(classification_report(test["label"], pred, labels=CLASSES,
                                    target_names=TARGET_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
