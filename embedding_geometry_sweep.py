import argparse
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks import load_arm
from modelling import split_train_test
from plot_style import set_publication_style, save_figure, style_axis, mark_split, REGIME_PALETTE, NEUTRAL_COLOURS
import embedding_geometry as eg

# Fixed to one market/arm/descriptor (S&P 500, raw, Graph2Vec, cosine, 2-D MDS) with EPOCH_LENGTHS and GRAPH_TYPES swept -- see docs/math/embedding-geometry.md.
INDEX_CODE = "SP500"
EPOCH_LENGTHS = [63, 132, 378]
GRAPH_TYPES = ["threshold", "mst", "tmfg"]
GRAPH_TYPE_LABELS = {"threshold": "Threshold", "mst": "MST", "tmfg": "TMFG"}
TRAIN_TEST_CUTOFF = "2019-12-31"
REGIME_NAMES = ["Calm", "Transitional", "Crisis"]

IMAGES_DIR = "images"
OUTPUTS_DIR = "outputs"
MODELS_DIR = f"{OUTPUTS_DIR}/models"
RESULTS_PATH = f"{OUTPUTS_DIR}/embedding_geometry_results.csv"
RESULTS_KEY_COLUMNS = ["epoch_length", "graph_type"]


# Images/results tag for one epoch_length -- graph_type is compared within each epoch_length's own figure set.
def _tag(epoch_length):
    return f"{INDEX_CODE}_{epoch_length}d"


# Frozen raw-arm Graph2Vec bundle tag for one (epoch_length, graph_type), matching modelling_sweep_raw.py's own tag formula.
def _bundle_tag(epoch_length, graph_type):
    return f"{INDEX_CODE}_raw_{epoch_length}d_{graph_type}"


# Save a figure into one epoch_length's images subfolder and close it. Dashboard diagnostics stay at 150dpi -- see docs/plot_style.md.
def _savefig(fig, tag, name):
    combo_dir = os.path.join(IMAGES_DIR, tag)
    os.makedirs(combo_dir, exist_ok=True)
    save_figure(fig, os.path.join(combo_dir, name), dpi=150)
    plt.close(fig)


# Create a fresh 1xn row of panels at the shared dashboard size.
def _new_axes_row(n):
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False, constrained_layout=True)
    return fig, axes[0]


# Set date-labelled ticks on an imshow axis's window-index axes (both x and y), at roughly interval_years spacing.
def _set_index_date_ticks(ax, dates, interval_years=2):
    years = pd.to_datetime(list(dates)).year
    positions, labels = [], []
    last_year = None
    for i, year in enumerate(years):
        if year != last_year and year % interval_years == 0:
            positions.append(i)
            labels.append(str(year))
            last_year = year
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)


# Draw one self-similarity heatmap panel with date ticks and the train/test split marked on both axes.
def _plot_selfsimilarity_panel(ax, D, dates, n_train, vmin, vmax):
    im = ax.imshow(D, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    _set_index_date_ticks(ax, dates)
    style_axis(ax)
    if 0 < n_train < len(dates):
        split_line = mark_split(ax, n_train - 0.5)
        ax.axhline(n_train - 0.5, color=split_line.get_color(), linestyle=split_line.get_linestyle(),
                   linewidth=split_line.get_linewidth(), zorder=split_line.get_zorder())
    return im


# Draw the self-similarity row: one panel per graph_type, one shared colour scale and colourbar for the whole figure.
def _plot_selfsimilarity_figure(distances_by_graph_type, dates_by_graph_type, n_train_by_graph_type, graph_types):
    fig, axes = _new_axes_row(len(graph_types))
    try:
        vmin = min(D.min() for D in distances_by_graph_type.values())
        vmax = max(D.max() for D in distances_by_graph_type.values())
        images = []
        for col, graph_type in enumerate(graph_types):
            D = distances_by_graph_type[graph_type]
            images.append(_plot_selfsimilarity_panel(axes[col], D, dates_by_graph_type[graph_type], n_train_by_graph_type[graph_type], vmin, vmax))
            axes[col].set_title(GRAPH_TYPE_LABELS[graph_type], fontweight="bold")
        fig.colorbar(images[-1], ax=list(axes), shrink=0.8, label="distance")
    except Exception:
        plt.close(fig)
        raise
    return fig


# Draw one 2-D MDS scatter panel coloured by the given colouring, with the graph_type name and the fit's stress in the title.
def _plot_mds_panel(ax, coords, colour_values, stress, colour_kind, cmap, vmin, vmax, graph_type):
    if colour_kind == "date":
        ax.plot(coords[:, 0], coords[:, 1], color=NEUTRAL_COLOURS["primary"], linewidth=0.5, alpha=0.3, zorder=1)

    if colour_kind == "regime":
        for regime_id, name, hexcolour in zip(range(3), REGIME_NAMES, REGIME_PALETTE):
            mask = colour_values == regime_id
            ax.scatter(coords[mask, 0], coords[mask, 1], color=hexcolour, s=20, alpha=0.8, label=name, zorder=2)
        sc = None
    else:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=colour_values, cmap=cmap, vmin=vmin, vmax=vmax, s=20, alpha=0.8, zorder=2)

    ax.set_xlabel("MDS Dimension 1")
    ax.set_ylabel("MDS Dimension 2")
    ax.set_title(f"{GRAPH_TYPE_LABELS[graph_type]} (stress={stress:.3f})", fontweight="bold")
    style_axis(ax, grid=True)
    return sc


# Draw one 2-D MDS figure for one colouring: one graph_type panel per column, sharing one colour scale.
def _plot_mds_figure(coords_by_graph_type, colour_values_by_graph_type, stress_by_graph_type, colour_kind, graph_types, colourbar_label=None):
    fig, axes = _new_axes_row(len(graph_types))

    try:
        if colour_kind == "regime":
            vmin = vmax = cmap = None
        else:
            all_values = np.concatenate([colour_values_by_graph_type[gt] for gt in graph_types])
            vmin, vmax, cmap = float(all_values.min()), float(all_values.max()), "viridis"

        last_sc = None
        for ax, graph_type in zip(axes, graph_types):
            sc = _plot_mds_panel(ax, coords_by_graph_type[graph_type], colour_values_by_graph_type[graph_type],
                                  stress_by_graph_type[graph_type], colour_kind, cmap, vmin, vmax, graph_type)
            if sc is not None:
                last_sc = sc

        if colour_kind == "regime":
            axes[0].legend(loc="best", frameon=True, facecolor="white", edgecolor="lightgray")
        elif last_sc is not None:
            fig.colorbar(last_sc, ax=list(axes), shrink=0.7, label=colourbar_label)
    except Exception:
        plt.close(fig)
        raise

    return fig


# Load the raw arm's train/test split once for one epoch_length; every graph_type within it reuses this split.
def _load_split(epoch_length):
    data = load_arm("raw", epoch_length, index_code=INDEX_CODE)
    train, test = split_train_test(data, cutoff=TRAIN_TEST_CUTOFF)
    dates = list(train["epoch_dates"]) + list(test["epoch_dates"])
    n_train = len(train["epoch_dates"])
    return train, test, dates, n_train


# Build one graph_type's Graph2Vec vector/distance/regime-labels from one epoch_length's shared raw-arm split.
def _graph2vec_entry(train, test, dates, n_train, epoch_length, graph_type):
    graph_key = "threshold_graphs" if graph_type == "threshold" else f"{graph_type}_threshold_graphs"
    train_graphs, test_graphs = train[graph_key], test[graph_key]
    edge_counts = [G.number_of_edges() for G in train_graphs + test_graphs]

    bundle, train_raw, test_raw, train_oov, test_oov = eg.load_graph2vec_raw(
        _bundle_tag(epoch_length, graph_type), train_graphs, test_graphs, MODELS_DIR)
    raw = np.concatenate([train_raw, test_raw])
    oov = train_oov + test_oov
    prepared = eg.prepare_graph2vec(raw)
    regime_labels = eg.raw_arm_regime_labels(bundle, test_raw, n_train)

    return {"raw": raw, "oov": oov, "dates": dates, "n_train": n_train,
            "edge_counts": edge_counts, "regime_labels": regime_labels,
            "distance": eg.distance_matrix(prepared)}


# Build one outputs/embedding_geometry_results.csv row for one (epoch_length, graph_type); stress is filled in once MDS is fit.
def _summary_row(entry, epoch_length, graph_type):
    mn, med, mx = eg.offdiag_distance_summary(entry["distance"])
    return {"index_code": INDEX_CODE, "epoch_length": epoch_length, "graph_type": graph_type,
            "descriptor": "graph2vec", "metric": "cosine", "arm": "raw",
            "n_windows": entry["distance"].shape[0], "n_train": entry["n_train"], "stress": None,
            "offdiag_dist_min": mn, "offdiag_dist_median": med, "offdiag_dist_max": mx,
            "mean_oov_rate": float(np.mean(entry["oov"])), "median_oov_rate": float(np.median(entry["oov"])),
            "mean_edge_count": float(np.mean(entry["edge_counts"])), "median_edge_count": float(np.median(entry["edge_counts"]))}


# Dedupe this run's rows by (epoch_length, graph_type) -- the only columns that vary, descriptor/metric/arm are constant -- then merge into RESULTS_PATH without clobbering other combos' existing rows.
def _write_results(rows):
    new_df = pd.DataFrame(rows)
    if not new_df.empty:
        new_df = new_df.drop_duplicates(subset=RESULTS_KEY_COLUMNS, keep="last").reset_index(drop=True)
    n_new = len(new_df)
    if os.path.exists(RESULTS_PATH):
        existing = pd.read_csv(RESULTS_PATH, keep_default_na=False, na_values=[])
        if new_df.empty:
            return existing
        if set(RESULTS_KEY_COLUMNS).issubset(existing.columns) and set(existing.columns) == set(new_df.columns):
            new_keys = set(map(tuple, new_df[RESULTS_KEY_COLUMNS].to_numpy()))
            keep = ~existing[RESULTS_KEY_COLUMNS].apply(tuple, axis=1).isin(new_keys)
            new_df = pd.concat([existing[keep], new_df], ignore_index=True)
        else:
            print(f"  existing {RESULTS_PATH} has a different schema (stale/pre-rewrite) -- overwriting rather than merging")
    elif new_df.empty:
        print(f"  nothing collected yet -- {RESULTS_PATH} not created")
        return new_df
    new_df.to_csv(RESULTS_PATH, index=False)
    print(f"wrote {len(new_df)} rows to {RESULTS_PATH} ({n_new} new/refreshed this run)")
    return new_df


# Build the distance-pickle path for one (epoch_length, graph_type), reusing the same tag as the frozen bundle it was derived from.
def _distance_pkl_path(epoch_length, graph_type):
    return f"{OUTPUTS_DIR}/{_bundle_tag(epoch_length, graph_type)}_graph2vec_cosine_distance.pkl"


# Run every graph_type for one already-loaded epoch_length: CSV rows, distance pickles, self-similarity figure, and MDS figures.
def _run_epoch_length(epoch_length, train, test, dates, n_train, graph_types):
    entries = {}
    failed = []
    for graph_type in graph_types:
        try:
            entries[graph_type] = _graph2vec_entry(train, test, dates, n_train, epoch_length, graph_type)
        except Exception as e:
            print(f"  epoch_length={epoch_length} graph_type={graph_type} FAILED: {INDEX_CODE} / raw: {e!r} -- skipped")
            failed.append(graph_type)

    if not entries:
        print(f"no graph_types available for {epoch_length}d -- nothing to write")
        return [], failed

    ok_graph_types = list(entries.keys())
    tag = _tag(epoch_length)
    rows_by_gt = {gt: _summary_row(entries[gt], epoch_length, gt) for gt in ok_graph_types}
    for gt in ok_graph_types:
        joblib.dump(entries[gt]["distance"], _distance_pkl_path(epoch_length, gt))

    distances = {gt: entries[gt]["distance"] for gt in ok_graph_types}
    dates_by_gt = {gt: entries[gt]["dates"] for gt in ok_graph_types}
    n_train_by_gt = {gt: entries[gt]["n_train"] for gt in ok_graph_types}
    fig1 = _plot_selfsimilarity_figure(distances, dates_by_gt, n_train_by_gt, ok_graph_types)
    _savefig(fig1, tag, "geometry_selfsimilarity.png")

    coords_by_gt, stress_by_gt = {}, {}
    for gt in ok_graph_types:
        coords_by_gt[gt], stress_by_gt[gt] = eg.fit_mds(entries[gt]["distance"])
        rows_by_gt[gt]["stress"] = float(stress_by_gt[gt])

    colourings = {
        "date": ({gt: np.arange(len(dates_by_gt[gt])) for gt in ok_graph_types}, "window index"),
        "regime": ({gt: entries[gt]["regime_labels"] for gt in ok_graph_types}, None),
        "edgecount": ({gt: np.array(entries[gt]["edge_counts"]) for gt in ok_graph_types}, "edge count"),
        "oov": ({gt: np.array(entries[gt]["oov"]) for gt in ok_graph_types}, "OOV rate"),
    }
    for colour_kind, (colour_values, label) in colourings.items():
        fig = _plot_mds_figure(coords_by_gt, colour_values, stress_by_gt, colour_kind, ok_graph_types, colourbar_label=label)
        _savefig(fig, tag, f"geometry_mds_graph2vec_{colour_kind}.png")

    return list(rows_by_gt.values()), failed


# Run the embedding-geometry pipeline (S&P 500, raw arm, Graph2Vec, cosine, 2-D MDS) across epoch lengths and network constructions, writing results after each epoch_length so a later failure can't discard earlier successes.
def main(epoch_lengths=EPOCH_LENGTHS, graph_types=GRAPH_TYPES):
    set_publication_style()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    written = pd.DataFrame()
    any_ok = False
    for epoch_length in epoch_lengths:
        train, test, dates, n_train = _load_split(epoch_length)
        rows, failed = _run_epoch_length(epoch_length, train, test, dates, n_train, graph_types)
        del train, test
        if rows:
            any_ok = True
            written = _write_results(rows)
        if failed:
            print(f"{epoch_length}d: {len(failed)} graph_type(s) failed: {failed}")

    if not any_ok:
        print("no epoch_length/graph_type combinations available -- nothing to write")
        return pd.DataFrame()

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embedding-geometry pipeline: S&P 500 / raw arm / Graph2Vec / cosine / 2D MDS, swept over epoch lengths and network constructions.")
    parser.add_argument("--epoch-lengths", nargs="+", type=int, default=EPOCH_LENGTHS, choices=EPOCH_LENGTHS,
                         help="Epoch lengths to run (default: all three).")
    parser.add_argument("--graph-types", nargs="+", default=GRAPH_TYPES, choices=GRAPH_TYPES,
                         help="Graph constructions to run (default: all three).")
    parser.add_argument("--quick", action="store_true",
                         help="Smoke-test the T=132/mst combo only -- reuses the frozen raw-arm bundle, fits nothing fresh.")
    args = parser.parse_args()

    main(epoch_lengths=[132] if args.quick else args.epoch_lengths,
         graph_types=["mst"] if args.quick else args.graph_types)
