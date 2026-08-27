import argparse
import os
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from plot_style import set_publication_style
import sweep_common as sc

from networks import load_arm
from modelling import load_vix, prepare_graph, split_train_test, fit_embed, transform_embed, empirical_transitions, validate_regime

K = sc.K
REGIME_NAMES = sc.REGIME_NAMES

ARMS = ["raw"]
EPOCH_LENGTHS = [63, 132, 378]
GRAPH_TYPES = ["threshold", "mst", "tmfg"]
INDEX_CODES = ["SP500", "Nikkei225", "FTSE350", "CSI300"]
TRAIN_TEST_CUTOFF = "2019-12-31"

IMAGES_DIR = "images"
RESULTS_PATH = "outputs/modelling_sweep_results_raw.csv"
MODELS_DIR = "outputs/models"
TRANSITIONS_PATH = "outputs/transition_matrices_raw.pkl"


# Run one grid point end to end and return its results.
def run_combo(index_code, arm, epoch_length, graph_type, vix, n_seeds=20):
    tag = f"{index_code}_{arm}_{epoch_length}d_{graph_type}"
    data = load_arm(arm, epoch_length, index_code=index_code)
    graph_key = "threshold_graphs" if graph_type == "threshold" else f"{graph_type}_threshold_graphs"

    train, test = split_train_test(data, cutoff=TRAIN_TEST_CUTOFF)
    train_graphs, test_graphs = train[graph_key], test[graph_key]
    train_mu, test_mu = train["mu_values"], test["mu_values"]
    train_dates, test_dates = train["epoch_dates"], test["epoch_dates"]

    X_train, g2v, scaler, pca, fit_diag = fit_embed(train_graphs)
    X_test, test_diag = transform_embed(test_graphs, g2v, scaler, pca)
    train_oov_rate = g2v.oov_rate([prepare_graph(G) for G in train_graphs])
    row = {
        "index_code": index_code, "arm": arm, "epoch_length": epoch_length, "graph_type": graph_type,
        "n_train_windows": len(train_graphs), "n_test_windows": len(test_graphs),
        "pca_dims": fit_diag["pca_dims"],
        "oov_rate_train_mean": float(np.mean(train_oov_rate)) if train_oov_rate else float("nan"),
        "oov_rate_test_mean": float(np.mean(test_diag["oov_rate"])) if test_diag["oov_rate"] else float("nan"),
    }

    results = sc.fit_and_rank_regimes(X_train, X_test, train_mu, n_seeds)

    transitions_entry = {}
    for method_name, method_result in results.items():
        row[f"{method_name}_crisis_mu"] = float(method_result["mean_mu"][method_result["crisis_id"]])
        row[f"{method_name}_n_train_clusters_used"] = int(np.sum(~np.isnan(method_result["mean_mu"])))
        if method_name == "hmm":
            for r, name in enumerate(REGIME_NAMES):
                row[f"hmm_dwell_{name.lower()}"] = float(method_result["dwell"][r])

        fig = sc.plot_regime_timeline(train_dates, train_mu, method_result["train_labels"], test_dates, test_mu,
                                       method_result["test_labels"], method_result["rank_of_cluster"])
        sc.savefig(fig, IMAGES_DIR, tag, f"{method_name}_timeline.png")

        if index_code == "SP500":
            res_train = validate_regime(method_result["train_labels"], train_dates, vix, 30.0, method_result["crisis_id"])
            res_test = validate_regime(method_result["test_labels"], test_dates, vix, 30.0, method_result["crisis_id"])
            row[f"{method_name}_train_point_precision"], row[f"{method_name}_train_point_recall"], row[f"{method_name}_train_point_f1"] = res_train["point"]
            row[f"{method_name}_test_point_precision"], row[f"{method_name}_test_point_recall"], row[f"{method_name}_test_point_f1"] = res_test["point"]
            idx_train = pd.DatetimeIndex(pd.to_datetime(train_dates))
            idx_test = pd.DatetimeIndex(pd.to_datetime(test_dates))
            fig = sc.plot_vix_overlay(idx_train, res_train, idx_test, res_test, epoch_length)
            sc.savefig(fig, IMAGES_DIR, tag, f"{method_name}_vix_overlay.png")

        emp_train, occ_train = empirical_transitions(method_result["train_labels"], K, method_result["order"])
        emp_test, occ_test = empirical_transitions(method_result["test_labels"], K, method_result["order"])
        fig = sc.plot_transition_comparison(emp_train, emp_test, occ_train, occ_test, f"{tag} ({method_name})")
        sc.savefig(fig, IMAGES_DIR, tag, f"{method_name}_transition_comparison.png")

        common_ranks = ~(np.isnan(np.diag(emp_train)) | np.isnan(np.diag(emp_test)))
        row[f"{method_name}_regime_persistence_gap"] = float(np.mean(np.abs(
            np.diag(emp_train)[common_ranks] - np.diag(emp_test)[common_ranks]))) if common_ranks.any() else float("nan")
        row[f"{method_name}_regimes_absent_test"] = int(((occ_train > 0) & (occ_test == 0)).sum())

        transitions_entry[method_name] = {"train": emp_train, "test": emp_test, "occ_train": occ_train, "occ_test": occ_test}

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({
        "g2v": g2v, "scaler": scaler, "pca": pca,
        "kmeans_model": results["kmeans"]["model"], "kmeans_order": results["kmeans"]["order"],
        "kmeans_crisis_id": results["kmeans"]["crisis_id"],
        "hmm_model": results["hmm"]["model"], "hmm_order": results["hmm"]["order"],
        "hmm_crisis_id": results["hmm"]["crisis_id"],
    }, os.path.join(MODELS_DIR, f"{tag}.joblib"))

    combo_key = (index_code, arm, epoch_length, graph_type)
    return row, combo_key, transitions_entry


# Run the raw-arm sweep grid and write the results to disk.
def main(index_codes=INDEX_CODES, arms=ARMS, epoch_lengths=EPOCH_LENGTHS, graph_types=GRAPH_TYPES, n_seeds=20):
    set_publication_style()
    vix = load_vix()

    rows = []
    transitions = {}
    failures = []
    total_combos = len(index_codes) * len(arms) * len(epoch_lengths) * len(graph_types)
    os.makedirs("outputs", exist_ok=True)
    for index_code in index_codes:
        for arm in arms:
            for epoch_length in epoch_lengths:
                for graph_type in graph_types:
                    print(f"running {index_code} / {arm} / {epoch_length}d / {graph_type} ...")
                    try:
                        row, combo_key, transitions_entry = run_combo(
                            index_code, arm, epoch_length, graph_type, vix, n_seeds=n_seeds)
                    except Exception as e:
                        print(f"combo FAILED: {index_code} / {arm} / {epoch_length}d / {graph_type}: {e!r}")
                        failures.append((index_code, arm, epoch_length, graph_type))
                        continue
                    rows.append(row)
                    transitions[combo_key] = transitions_entry

                    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
                    with open(TRANSITIONS_PATH, "wb") as f:
                        pickle.dump(transitions, f)

    df = pd.DataFrame(rows)
    print(f"wrote {len(df)} rows to {RESULTS_PATH} and {len(transitions)} entries to {TRANSITIONS_PATH}")
    if failures:
        print(f"{len(failures)}/{total_combos} combos failed: {failures}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep the index_code x epoch_length x graph_type grid for the raw arm only.")
    parser.add_argument("--n-seeds", type=int, default=20, help="HMM restarts for cluster_hmm (default: 20)")
    parser.add_argument("--quick", action="store_true", help="Smoke-test a single (SP500, raw, 132, mst) combo with low seed counts")
    args = parser.parse_args()

    if args.quick:
        main(index_codes=["SP500"], arms=["raw"], epoch_lengths=[132], graph_types=["mst"], n_seeds=min(args.n_seeds, 3))
    else:
        main(n_seeds=args.n_seeds)