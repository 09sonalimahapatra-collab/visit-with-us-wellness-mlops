"""Generate compact EDA charts used in the submitted notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .config import PROJECT_ROOT, RAW_DATA_PATH, TARGET_COLUMN
except ImportError:  # pragma: no cover - supports direct execution
    from config import PROJECT_ROOT, RAW_DATA_PATH, TARGET_COLUMN


ASSET_DIR = PROJECT_ROOT / "notebook_assets"


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RAW_DATA_PATH)

    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Did not purchase", "Purchased"]
    colors = ["#8FB8DE", "#0B6E8A"]
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    bars = axis.bar(labels, counts.values, color=colors, width=0.58)
    axis.set_ylabel("Customers")
    axis.set_title("Target distribution: Wellness Package purchase")
    axis.set_ylim(0, counts.max() * 1.18)
    for bar, count in zip(bars, counts.values):
        axis.annotate(
            f"{count:,}\n({count / len(frame):.1%})",
            (bar.get_x() + bar.get_width() / 2, count),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    figure.tight_layout()
    figure.savefig(ASSET_DIR / "target_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    conversion = (
        frame.groupby("ProductPitched")[TARGET_COLUMN]
        .mean()
        .sort_values(ascending=True)
        .mul(100)
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    bars = axis.barh(conversion.index, conversion.values, color="#4C9F70")
    axis.set_xlabel("Purchase rate (%)")
    axis.set_title("Observed purchase rate by product pitched")
    for bar, value in zip(bars, conversion.values):
        axis.annotate(f"{value:.1f}%", (value, bar.get_y() + bar.get_height() / 2), va="center", xytext=(4, 0), textcoords="offset points")
    figure.tight_layout()
    figure.savefig(ASSET_DIR / "purchase_rate_by_product.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"EDA assets saved in: {ASSET_DIR}")


if __name__ == "__main__":
    main()
