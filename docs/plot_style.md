# Figure style: single visual language via `plot_style.py`

Implemented in `plot_style.py`. This module is the single source of truth for figure styling across the project — fonts, colours, spines, ticks, export settings — and is the styling backend for every sweep-script and walkthrough-notebook figure (see "Scope"). No figure in scope should redefine any of these locally; a figure that wants to deviate is a signal to extend this module, not to inline an exception.

**Scope**: every figure the three regime-detection sweep scripts (`modelling_sweep_raw.py`, `eigencentrality_sweep_raw.py`, `spectral_sweep_raw.py`) and their three walkthrough notebooks (`modelling_walkthrough.ipynb`, `eigencentrality_walkthrough.ipynb`, `spectral_walkthrough.ipynb`) produce goes through this module for styling — `sweep_common.plot_regime_timeline`/`plot_vix_overlay`/`plot_transition_comparison` (one shared implementation the three sweep scripts import, not three local copies — see `CLAUDE.md`'s Library modules) and the matching notebook `plot_regime_timeline`/`plot_vix_overlay` functions (each notebook keeps its own local copy, not shared with `sweep_common.py`) all call `set_publication_style`/`style_axis`/`format_date_axis`/`mark_split`/`annotate_events`, using `plot_style.REGIME_PALETTE`/`EPOCH_LENGTH_COLOURS`/`CRISIS_EVENTS` rather than any local copy. Persistence differs by design: the sweep scripts also call `save_figure` (via `sweep_common.savefig`) to write every combo's figures to `images/`; the walkthrough notebooks are display-only (`plt.show()`, no `save_figure` call) since they're meant for interactive/narrative reading, not artifact generation — the sweep scripts already persist the full grid. `data_visualisation.ipynb` is the one exception outside this scope entirely — it's an independent notebook (network-layout animations, not a sweep/walkthrough figure) that deliberately resets to matplotlib's default style and uses its own seaborn/`jet` colour choices instead.

**`embedding_geometry_sweep.py`** (the 4th sweep script) is a partial consumer: its self-similarity heatmaps and 2-D MDS scatter panels are a different plot vocabulary entirely (no `sweep_common.plot_regime_timeline`/`plot_vix_overlay` pattern applies), but they still build on this module's primitives directly — `set_publication_style`, `style_axis`, `save_figure` (at the same `dpi=150` per-combo convention as the other three sweep scripts), and `REGIME_PALETTE` for its regime colouring, plus `mark_split` for the self-similarity heatmaps' train/test split lines. See [`math/embedding-geometry.md`](math/embedding-geometry.md) for what its figures show.

**The four `data_download_*.ipynb` notebooks** (`data_download_sp500.ipynb`, `data_download_nikkei225.ipynb`, `data_download_ftse350.ipynb`, `data_download_csi300.ipynb`) are simpler consumers: each ends with diagnostic cells (a mean-adjusted-close line, an all-constituents log-scale overlay, an all-constituents log-return overlay, a halted-stock plot) calling `set_publication_style()`/`style_axis(ax, grid=True)`/`format_date_axis()`/`NEUTRAL_COLOURS["primary"]` directly — no regime shading, no `mark_split`/`annotate_events`/`REGIME_PALETTE`/`save_figure`, since these are pre-modelling sanity-check plots with no regime labels or train/test split to show yet. `plt.show()`-only, matching the walkthrough notebooks' display-only convention.

## Design philosophy

Scientific clarity over decoration, consistency over novelty, readability at journal size, information density without clutter — modelled on physics/complex-systems journal figure conventions (EPJ Data Science / Nature Communications / PNAS / Physical Review E style). Where a styling choice would improve aesthetics but reduce clarity, clarity wins. No dashboard aesthetics, no default Matplotlib styling, no unnecessary annotation.

## Dimensions

- Single-column figures: `SINGLE_COL_WIDTH = 3.35` inches (8.5 cm).
- Double-column figures: `DOUBLE_COL_WIDTH = 6.9` inches (17.5 cm).
- Typical height: 3–5 inches; `figsize(width="single"|"double", height=3.5)` returns the `(width, height)` tuple for `plt.subplots`.
- Export: `dpi=600`, `bbox_inches="tight"`, `transparent=False` — enforced by `save_figure`, and also set as the `savefig.*` rcParams defaults by `set_publication_style`, so even a bare `fig.savefig(...)` call inherits them. Every sweep script's per-combo diagnostic images are the one deliberate exception: they call `save_figure(fig, path, dpi=150)` explicitly, since a 600dpi export of every regime-timeline/VIX-overlay/transition-comparison/self-similarity/MDS image across the full multi-market grid (hundreds of PNGs under the gitignored `images/` directory) would multiply render time and disk use for plots nobody reads at print resolution. Only figures actually destined for the manuscript should rely on `save_figure`'s 600dpi default. The walkthrough notebooks don't call `save_figure` at all (see "Scope" above) — the dpi override is moot for them.

## Colour system

`get_colour_palette()` returns the manuscript's fixed colours as `{"regime": {...}, "neutral": {...}}` — nothing else in the manuscript should hardcode a hex value.

**Regime colours** (must stay identical across every regime timeline, embedding scatter, transition-matrix annotation, and network visualisation in the paper):

| Regime | Colour | Hex |
|---|---|---|
| Calm | green | `#2ca02c` |
| Transition | blue | `#1f77b4` |
| Crisis | red | `#d62728` |

These map onto the pipeline's `REGIME_NAMES = ["Calm", "Transitional", "Crisis"]` (see [`math/regime-detection.md`](math/regime-detection.md)); the dict keys here use `"transition"` rather than `"transitional"` per this style guide's own vocabulary. `REGIME_PALETTE = [REGIME_COLOURS["calm"], REGIME_COLOURS["transition"], REGIME_COLOURS["crisis"]]` is the same three colours as a plain list, ordered to match `REGIME_NAMES`/`rank_of_cluster` indexing — every regime-shading call site indexes into `REGIME_PALETTE[rank]`, not the dict, since `fit_regime_order`/`apply_regime_order` (see [`math/regime-detection.md`](math/regime-detection.md)) hand back integer ranks, not colour names.

**Neutral colours**: `primary`/`text` (black, `#000000`) for main lines and text, `reference` (medium grey, `#7f7f7f`) for reference lines, `ci` (light grey, `#d9d9d9`) for confidence intervals/background shading, `background` (very light grey, `#f5f5f5`) for background panels.

**Two more shared constants**:

- **`CRISIS_EVENTS`** — the fixed `{date_str: label}` dict of historical crisis dates (Dot-com/GFC/Euro Debt/COVID/Rate Shock) annotated on every VIX overlay.
- **`EPOCH_LENGTH_COLOURS`** — `{378: "blue", 132: "green", 63: "purple"}`, the VIX curve's colour keyed by epoch length. Named CSS colours, not hex — it labels a *timescale* on an already-restrained figure, not part of the paper's fixed regime/neutral colour vocabulary above.

## Functions

- **`set_publication_style()`** — sets all global `matplotlib.rcParams` in one call: `DejaVu Sans` throughout (title 12pt, axis labels 11pt, tick/legend 9pt), `axes.linewidth=1.0`, outward ticks (`length=4, width=1`), top/right spines off, no grid by default (`grid.alpha=0.15` if enabled per-axis), white figure/savefig background, `dpi=600`/`bbox_inches="tight"` savefig defaults. Call once, at the top of any script or notebook that produces figures in scope, before creating any axes.
- **`get_colour_palette()`** — returns the colour dict above; import this (or the `REGIME_PALETTE`/`NEUTRAL_COLOURS` constants directly) rather than re-typing hex codes at a call site.
- **`figsize(width="single", height=3.5)`** — returns `(SINGLE_COL_WIDTH or DOUBLE_COL_WIDTH, height)` for `plt.subplots(figsize=...)`. Not used by the sweep/walkthrough figures, which keep their existing `figsize=(12, 4)` dashboard dimensions (wide time series, not single/double-column manuscript panels) — this helper is for actual manuscript figures.
- **`style_axis(ax, grid=False)`** — removes top/right spines, sets left/bottom spine width and outward tick length/width, and turns the faint (`alpha=0.15`) grid on or off. Call on every axis after creating it, even ones `set_publication_style()`'s rcParams already mostly cover, since rcParams don't reach per-axis spine visibility reliably across all Matplotlib backends.
- **`save_figure(fig, filename, dpi=600)`** — the one call site that should ever write a figure to disk: fixed `dpi`, `bbox_inches="tight"`, `transparent=False`. See "Dimensions" above for why the sweep/walkthrough call sites override `dpi=150`.
- **`add_panel_label(ax, label, x=-0.15, y=1.05)`** — places a bold 14pt label (e.g. `"A"`) just outside an axis's top-left corner in axes-fraction coordinates, for multi-panel figures. Pass the bare letter (`"A"`, not `"(A)"`); parenthesise at the call site if the target format wants parentheses.
- **`shade_periods(ax, intervals, colour, alpha=0.2, zorder=1)`** — draws `ax.axvspan` for a list of `(start, end)` x-intervals with one consistent colour/alpha (semi-transparent, never opaque). Regime-timeline shading calls this once per regime per train/test split (colour from `REGIME_PALETTE`); VIX-overlay crisis shading calls it once per train/test split (colour `REGIME_PALETTE[2]`, the crisis colour).
- **`format_date_axis(ax, interval_years=2)`** — sets biennial year ticks (`mdates.YearLocator`/`DateFormatter("%Y")`) for any date-indexed figure.
- **`mark_split(ax, split_date, colour="#333333", label="Train/test split")`** — draws the dashed vertical train/test-split marker and returns its `Line2D` legend handle, so the call site can just append the return value to its `handles` list rather than building the line and the matching legend entry separately.
- **`annotate_events(ax, events, x_min, x_max, colour="#444444", fontsize=9, y_frac=0.97)`** — draws `CRISIS_EVENTS`-style dotted vertical lines with rotated labels, skipping any event outside `[x_min, x_max]`. Every VIX overlay calls this with `events=plot_style.CRISIS_EVENTS`.

## What this module deliberately does not do

No `style_legend` helper is provided: legends should sit at `upper left`/`upper right`/outside the axes with a light or absent box (9pt text, per the style guide) — a call-site decision (a two-line `ax.legend(...)` call), not complex enough to warrant a wrapper.

Network-visualisation-specific conventions (deterministic layouts, fixed node coordinates for comparison, edge-alpha-by-weight, small node size, sparse labelling) and heatmap/matrix conventions (`viridis`/`cividis`/`magma` for unsigned quantities, zero-centred diverging maps for signed/correlation quantities, always a colourbar) are call-site responsibilities layered on top of `style_axis`/`save_figure`, not separate functions here — they're compositional rather than parametrisable into one helper each.

## Adoption status

Fully adopted by all three regime-detection sweep scripts and all three walkthrough notebooks (see "Scope" above), across all four markets those sweep scripts now cover. The four `data_download_*.ipynb` notebooks are also full consumers for their diagnostic plots (see above), and `embedding_geometry_sweep.py` is a partial consumer (see above). `data_visualisation.ipynb`'s network-layout animation and any future manuscript-only figures (using `figsize`/`add_panel_label`/the 600dpi `save_figure` default) remain the module's only unaddressed consumers.
