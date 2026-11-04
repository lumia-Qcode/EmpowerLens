"""Class-distribution figure for Chapter 5, section 5.2.6.

Counts are read from Annotated_data.csv rather than hard-coded, so the figure
cannot drift from the data. Writes a vector PDF for LaTeX and a PNG to preview.

    venv\\Scripts\\python.exe -m src.plot_class_distribution
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import LABEL_CANON  # noqa: F401  (import proves labels are mapped)

RAW = Path("Annotated_data.csv")
OUT = Path("planning/figures")
NO_DIST = "no distortion"


def main() -> int:
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    counts = df["Dominant Distortion"].astype(str).str.strip().value_counts()

    # No Distortion is plotted apart from the ten: it is the contrast the
    # rebalancing argument rests on, not one of the distortion classes.
    nd = [(k, v) for k, v in counts.items() if k.lower() == NO_DIST]
    dist = [(k, v) for k, v in counts.items() if k.lower() != NO_DIST]
    dist.sort(key=lambda kv: kv[1])
    ordered = dist + nd

    labels = [k for k, _ in ordered]
    values = [v for _, v in ordered]
    colours = ["#4C72B0"] * len(dist) + ["#B0B0B0"] * len(nd)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.barh(labels, values, color=colours, height=0.72)
    for bar, v in zip(bars, values):
        ax.text(v + 12, bar.get_y() + bar.get_height() / 2, str(v),
                va="center", fontsize=8)

    ax.set_xlabel("Rows in the benchmark corpus", fontsize=9)
    ax.set_xlim(0, max(values) * 1.12)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"class_distribution.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"total rows: {sum(values)}")
    print(f"largest distortion : {dist[-1][0]} ({dist[-1][1]})")
    print(f"smallest distortion: {dist[0][0]} ({dist[0][1]})")
    print(f"ratio within the ten distortions: {dist[-1][1] / dist[0][1]:.2f}x")
    if nd:
        print(f"no distortion vs largest distortion: {nd[0][1] / dist[-1][1]:.2f}x")
    print(f"wrote -> {OUT / 'class_distribution.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
