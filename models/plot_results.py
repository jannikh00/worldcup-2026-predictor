"""
plot_results.py
===============
Render the figure README.md leads with: accuracy by evaluation protocol, and
the fold-to-fold spread that swallows the gap between models.

The numbers are NOT hardcoded. This imports the protocol functions from
evaluate.py and calls them, so the figure cannot drift from the evaluation it
illustrates - the same objection rescrape_missing.py avoids by reusing
scrape_squads' own functions. evaluate.py prints as it goes, so its output is
captured and discarded here.

Two design choices worth naming:

  * DOTS, NOT BARS. Every accuracy sits between 0.42 and 0.57. A bar chart must
    start at zero, which would compress all sixteen values into the top quarter
    of the plot; truncating the axis to fix that is the classic misleading bar
    chart. Position-encoded dots carry the same values honestly at full
    resolution.
  * NO VALUE ON EVERY DOT. Sixteen numbers on one panel is noise. The x-axis
    and gridlines carry the left panel; the README table beside the figure is
    the accessible table view. Only the four means in the right panel - where
    the argument actually is - are direct-labelled.

Run:  python models/plot_results.py [--out-dir DIR]
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")               # headless: no display needed, works in CI
import matplotlib.pyplot as plt     # noqa: E402
import pandas as pd                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (              # noqa: E402
    load, make_models, protocol_a, protocol_b, protocol_c, protocol_d,
)

DOCS = Path(__file__).resolve().parents[1] / "docs"
CUTOFF = pd.Timestamp("2025-01-01")

# Protocol labels, in the order the argument is made. Plain ASCII on purpose:
# an arrow glyph is missing from Helvetica Neue and renders as a tofu box on
# macOS while working fine on the Linux runner, so the figure would differ
# between the machine that commits it and CI.
PROTOCOLS = [
    ("A", "mirror, then random split\n(contaminated)"),
    ("B", "random split,\nmirror train only"),
    ("C", "chronological split,\nmirror train only"),
    ("D", "forward-chaining CV\n(mean of 5 folds)"),
]

# --- palette -------------------------------------------------------------
# Categorical slots 1-4, assigned in the order make_models() returns them, so a
# model keeps its hue across both panels. Both sets pass the adjacent-pair CVD,
# normal-vision, lightness-band and chroma gates against their own surface;
# light mode's aqua and yellow sit below 3:1 contrast, which is why the README
# ships the full table beside the figure.
LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "band": "#f4f3ef",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
}
DARK = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
    "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
    "band": "#222220",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
}


def collect():
    """Run all four protocols and return the numbers, without their printing."""
    df = load()
    with contextlib.redirect_stdout(io.StringIO()):
        a, shared = protocol_a(df)
        b = protocol_b(df)
        c = protocol_c(df, CUTOFF)
        d = protocol_d(df)
    models = list(make_models())
    acc = {
        "A": {m: a[m]["accuracy"] for m in models},
        "B": {m: b[m]["accuracy"] for m in models},
        "C": {m: c[m]["accuracy"] for m in models},
        "D": {m: d[m][0] for m in models if m in d},
    }
    spread = {m: d[m] for m in models if m in d}      # name -> (mean, std)
    return models, acc, spread, shared


def xlimits(acc, spread):
    """One x-range for both panels.

    The panels sit side by side, so a reader compares dot positions across them.
    Letting each autoscale would put the same accuracy at two different
    horizontal positions - the same misreading a dual-axis chart invites.
    """
    vals = [v for row in acc.values() for v in row.values()]
    vals += [m - s for m, s in spread.values()]
    vals += [m + s for m, s in spread.values()]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.09
    return lo - pad, hi + pad


def draw(models, acc, spread, shared, theme, path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.0, 5.3), gridspec_kw={"width_ratios": [1.7, 1]}
    )
    # Explicit margins rather than tight_layout: the title and subtitle are
    # placed in figure coordinates, and tight_layout would reflow the axes out
    # from under them.
    fig.subplots_adjust(left=0.155, right=0.985, top=0.70, bottom=0.175,
                        wspace=0.42)

    colour = dict(zip(models, theme["series"]))
    # Sub-row offsets inside each protocol group, first model on top.
    offsets = [0.255, 0.085, -0.085, -0.255]

    # ---------------------------------------------------------------- left
    for i, (key, _) in enumerate(PROTOCOLS):
        if i % 2 == 0:                                   # quiet zebra banding
            ax1.axhspan(i - 0.5, i + 0.5, color=theme["band"], zorder=0, lw=0)
        for m, off in zip(models, offsets):
            if m not in acc[key]:
                continue
            ax1.plot(
                acc[key][m], i + off, "o",
                markersize=9, color=colour[m],
                markeredgecolor=theme["surface"], markeredgewidth=2,  # surface ring
                zorder=3, clip_on=False,
            )

    ax1.set_yticks(range(len(PROTOCOLS)))
    ax1.set_yticklabels([f"{k}  —  {lab}" for k, lab in PROTOCOLS],
                        fontsize=8.5, color=theme["ink2"], linespacing=1.5)
    ax1.set_ylim(len(PROTOCOLS) - 0.5, -0.5)             # A at the top
    ax1.set_xlabel("Three-class accuracy", fontsize=8.5, color=theme["ink2"],
                   labelpad=8)
    ax1.set_title("Accuracy by protocol", fontsize=10.5, color=theme["ink"],
                  loc="left", pad=12, fontweight="bold")

    # ---------------------------------------------------------------- right
    order = [m for m in models if m in spread]
    for j, m in enumerate(order):
        mean, std = spread[m]
        ax2.errorbar(
            mean, j, xerr=std, fmt="o",
            markersize=9, color=colour[m],
            markeredgecolor=theme["surface"], markeredgewidth=2,
            ecolor=colour[m], elinewidth=2, capsize=4, capthick=2,
            zorder=3, clip_on=False,
        )
        # Selective direct labels: only the four values the argument rests on.
        ax2.annotate(f"{mean:.3f}", (mean, j), textcoords="offset points",
                     xytext=(0, 13), ha="center", fontsize=8.5,
                     color=theme["ink"], zorder=4)

    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(order, fontsize=8.5, color=theme["ink2"])
    ax2.set_ylim(len(order) - 0.5, -0.5)
    ax2.set_xlabel("Accuracy, mean ± 1 SD across folds", fontsize=8.5,
                   color=theme["ink2"], labelpad=8)
    ax2.set_title("Protocol D — fold-to-fold spread", fontsize=10.5,
                  color=theme["ink"], loc="left", pad=12, fontweight="bold")

    # ---------------------------------------------------------------- chrome
    xlim = xlimits(acc, spread)
    for ax in (ax1, ax2):
        ax.set_xlim(*xlim)                    # shared scale across both panels
        ax.grid(axis="x", color=theme["grid"], linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["axis"])
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(axis="x", colors=theme["muted"], labelsize=8.5, length=0)
        ax.tick_params(axis="y", length=0)

    # Legend carries identity, so nothing depends on colour alone.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                   color=colour[m], markeredgecolor=theme["surface"],
                   markeredgewidth=2, label=m)
        for m in models
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(models),
        frameon=False, fontsize=8.5, labelcolor=theme["ink2"],
        bbox_to_anchor=(0.5, 0.005), handletextpad=0.4, columnspacing=1.8,
    )

    fig.text(
        0.012, 0.955,
        "The evaluation protocol moves the number more than the model does",
        fontsize=13, color=theme["ink"], ha="left", va="top", fontweight="bold",
    )
    fig.text(
        0.012, 0.878,
        f"Protocol A has its mirrored twin in training for {shared} of its test matches, yet scores no higher: "
        "mirroring the test set erases\nhome advantage. C is the deployment-shaped number; D shows the fold-to-fold "
        "spread that a single split hides.",
        fontsize=9.5, color=theme["ink2"], ha="left", va="top", linespacing=1.7,
    )

    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DOCS,
                    help="where to write the PNGs (default: docs/)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    models, acc, spread, shared = collect()
    draw(models, acc, spread, shared, LIGHT,
         args.out_dir / "protocol-comparison.png")
    draw(models, acc, spread, shared, DARK,
         args.out_dir / "protocol-comparison-dark.png")


if __name__ == "__main__":
    main()
