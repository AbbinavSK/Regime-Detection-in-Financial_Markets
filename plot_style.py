import matplotlib
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.lines import Line2D

SINGLE_COL_WIDTH = 3.35
DOUBLE_COL_WIDTH = 6.9

REGIME_COLOURS = {"calm": "#2ca02c", "transition": "#1f77b4", "crisis": "#d62728"}
NEUTRAL_COLOURS = {
    "primary": "#000000",
    "reference": "#7f7f7f",
    "ci": "#d9d9d9",
    "background": "#f5f5f5",
    "text": "#000000",
}
# Ordered to match REGIME_NAMES = ["Calm", "Transitional", "Crisis"] (docs/math/regime-detection.md) so REGIME_PALETTE[rank_of_cluster[label]] gives the right colour.
REGIME_PALETTE = [REGIME_COLOURS["calm"], REGIME_COLOURS["transition"], REGIME_COLOURS["crisis"]]

# Shared across every sweep script's/walkthrough's VIX overlay so the annotated dates stay identical.
CRISIS_EVENTS = {
    "2000-03-01": "Dot-com", "2008-09-15": "GFC", "2011-08-01": "Euro Debt",
    "2020-03-01": "COVID", "2022-06-01": "Rate Shock",
}
# VIX-curve colour keyed by epoch length, shared across every sweep script's/walkthrough's VIX overlay.
EPOCH_LENGTH_COLOURS = {378: "blue", 132: "green", 63: "purple"}


# Set global rcParams so every manuscript figure shares one visual language.
def set_publication_style():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.15,
        "grid.color": "#000000",
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


# Return the manuscript's fixed colour vocabulary (regime + neutral) as one nested dict.
def get_colour_palette():
    return {"regime": dict(REGIME_COLOURS), "neutral": dict(NEUTRAL_COLOURS)}


# Return a (width, height) figsize tuple in inches for a single- or double-column figure.
def figsize(width="single", height=3.5):
    col_width = SINGLE_COL_WIDTH if width == "single" else DOUBLE_COL_WIDTH
    return (col_width, height)


# Apply the manuscript's spine/tick/grid convention to a single axis.
def style_axis(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(direction="out", length=4, width=1.0)
    if grid:
        ax.grid(True, alpha=0.15, color="black", linewidth=0.6)
    else:
        ax.grid(False)
    return ax


# Save a figure at publication resolution with the manuscript's fixed export settings.
def save_figure(fig, filename, dpi=600):
    fig.savefig(filename, dpi=dpi, bbox_inches="tight", transparent=False)


# Add a bold panel label (e.g. "A") outside an axis's top-left corner, journal-style.
def add_panel_label(ax, label, x=-0.15, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="right")


# Shade a list of (start, end) x-intervals on an axis with one consistent colour/alpha.
def shade_periods(ax, intervals, colour, alpha=0.2, zorder=1):
    for start, end in intervals:
        ax.axvspan(start, end, color=colour, alpha=alpha, linewidth=0, zorder=zorder)


# Format an axis's x-axis as evenly-spaced year ticks, matching every date-indexed figure.
def format_date_axis(ax, interval_years=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(interval_years))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# Draw the dashed train/test split marker at split_date and return its legend handle.
def mark_split(ax, split_date, colour="#333333", label="Train/test split"):
    ax.axvline(split_date, color=colour, linestyle="--", linewidth=1.5, zorder=5)
    return Line2D([0], [0], color=colour, linestyle="--", linewidth=1.5, label=label)


# Annotate a {date_str: label} dict (e.g. CRISIS_EVENTS) as dotted vertical lines within [x_min, x_max].
def annotate_events(ax, events, x_min, x_max, colour="#444444", fontsize=9, y_frac=0.97):
    y_top = ax.get_ylim()[1]
    for date_str, label in events.items():
        event_date = pd.to_datetime(date_str)
        if x_min <= event_date <= x_max:
            ax.axvline(event_date, color=colour, linestyle=":", linewidth=1.3, alpha=0.85, zorder=4)
            ax.text(event_date, y_top * y_frac, label, rotation=90, fontsize=fontsize, color=colour, va="top", ha="center")