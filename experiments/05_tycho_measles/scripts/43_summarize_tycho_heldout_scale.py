from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd


DATA_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_NAME = os.environ.get("TYCHO_HELDOUT_SPLIT_NAME", "prevaccine_1959_1960")


def recovery_path(method_suffix):
    if SPLIT_NAME == "transition_1967_1968":
        return DATA_DIR / f"tycho_heldout_recovery_{method_suffix}.csv"
    return DATA_DIR / f"tycho_heldout_recovery_{SPLIT_NAME}_{method_suffix}.csv"


def summarize_method(method, method_suffix):
    recovery = pd.read_csv(recovery_path(method_suffix))
    test = recovery.loc[recovery["split"] == "test"].copy()
    observed = test["cases"].to_numpy(dtype=float)
    expected = test["expected_cases_mean"].to_numpy(dtype=float)
    error = expected - observed

    return {
        "split_name": SPLIT_NAME,
        "method": method,
        "n_test": len(test),
        "test_mean_cases": observed.mean(),
        "test_median_cases": np.median(observed),
        "test_max_cases": observed.max(),
        "test_zero_fraction": np.mean(observed == 0),
        "rmse_count_mean": np.sqrt(np.mean(error**2)),
        "mae_count_mean": np.mean(np.abs(error)),
        "rmse_over_mean_cases": np.sqrt(np.mean(error**2)) / observed.mean(),
        "mae_over_mean_cases": np.mean(np.abs(error)) / observed.mean(),
        "log1p_rmse": np.sqrt(np.mean((np.log1p(expected) - np.log1p(observed)) ** 2)),
        "log1p_mae": np.mean(np.abs(np.log1p(expected) - np.log1p(observed))),
        "coverage_95": test["covered_95"].mean(),
    }


def main():
    rows = [
        summarize_method("HSGP VI NB", "hsgp"),
    ]
    for latent_dim in [60, 120, 240]:
        compressed_path = recovery_path(f"compressed_z{latent_dim}")
        if compressed_path.exists():
            rows.append(
                summarize_method(
                    f"Compressed VI NB, r={latent_dim}",
                    f"compressed_z{latent_dim}",
                )
            )
    for latent_dim in [60, 120, 240]:
        official_path = recovery_path(f"official_deeprv_z{latent_dim}")
        if official_path.exists():
            rows.append(
                summarize_method(
                    f"Official DeepRV VI NB, r={latent_dim}",
                    f"official_deeprv_z{latent_dim}",
                )
            )
        official_coeff_path = recovery_path(f"official_deeprv_coeff_z{latent_dim}")
        if official_coeff_path.exists():
            rows.append(
                summarize_method(
                    f"Official coefficient DeepRV VI NB, r={latent_dim}",
                    f"official_deeprv_coeff_z{latent_dim}",
                )
            )
    summary = pd.DataFrame(rows)
    output_path = DATA_DIR / f"tycho_heldout_scale_adjusted_metrics_{SPLIT_NAME}.csv"
    summary.round(6).to_csv(output_path, index=False)
    print(summary.round(4).to_string(index=False))
    print(output_path)


if __name__ == "__main__":
    main()
