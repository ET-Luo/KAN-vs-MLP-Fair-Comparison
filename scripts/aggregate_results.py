#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = PROJECT_ROOT / "results" / "metrics" / "all_experiments.csv"
SUMMARY_CSV = PROJECT_ROOT / "results" / "metrics" / "summary.csv"


def main() -> None:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)

    group_cols = ["dataset", "param_budget", "model"]
    summary = (
        df.groupby(group_cols)
        .agg(
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            best_val_accuracy_mean=("best_val_accuracy", "mean"),
            best_val_accuracy_std=("best_val_accuracy", "std"),
            epochs_trained_mean=("epochs_trained", "mean"),
            training_time_hours_mean=("training_time_hours", "mean"),
        )
        .reset_index()
    )

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"[INFO] Summary saved to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
