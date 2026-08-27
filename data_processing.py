from networks import load_logreturns, rolling_correlations, build_arm_networks, save_arm, ARM_THRESHOLDS

INDICES = ["SP500", "Nikkei225", "FTSE350", "CSI300"]
EPOCH_CONFIGS = [(378, 22), (132, 10), (63, 5)]

for index_code in INDICES:
    df_logreturns = load_logreturns(f"data/{index_code}_AdjClose_Cleaned.csv")

    for epoch_length, epoch_shift in EPOCH_CONFIGS:
        correlation_matrices, epoch_dates, stock_names = rolling_correlations(df_logreturns, epoch_length, epoch_shift)
        print(f"{index_code} T={epoch_length}: {correlation_matrices[0].shape} correlation matrix, {len(correlation_matrices)} windows")

        raw = build_arm_networks(correlation_matrices, "raw", stock_names, epoch_length, epoch_dates, threshold=ARM_THRESHOLDS["raw"], index_code=index_code)
        save_arm(raw)