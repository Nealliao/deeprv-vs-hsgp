from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    hsgp = pd.read_csv(DATA_DIR / "tycho_hsgp_vi_metrics_1935_1968_negative_binomial.csv")
    hsgp["comparison_method"] = "HSGP VI NB"
    hsgp["latent_dim"] = ""
    hsgp["explained_prior_variance"] = ""
    hsgp["D_mean_vs_hsgp_vi"] = 0.0
    hsgp["D_sd_vs_hsgp_vi"] = 0.0
    hsgp["posterior_mean_correlation_vs_hsgp_vi"] = 1.0

    compressed = []
    for latent_dim in [120, 240]:
        table = pd.read_csv(
            DATA_DIR / f"tycho_compressed_decoder_vi_metrics_1935_1968_z{latent_dim}.csv"
        )
        table["comparison_method"] = f"Compressed VI NB z={latent_dim}"
        compressed.append(table)
    compressed = pd.concat(compressed, ignore_index=True)

    columns = [
        "comparison_method",
        "latent_dim",
        "explained_prior_variance",
        "runtime_seconds",
        "rmse_count_mean",
        "mae_count_mean",
        "predictive_count_interval_coverage_95",
        "mean_latent_sd",
        "D_mean_vs_hsgp_vi",
        "D_sd_vs_hsgp_vi",
        "posterior_mean_correlation_vs_hsgp_vi",
        "nb_concentration_mean",
    ]
    comparison = pd.concat([hsgp[columns], compressed[columns]], ignore_index=True)
    hsgp_runtime = float(comparison.loc[0, "runtime_seconds"])
    comparison["speedup_vs_hsgp_vi"] = hsgp_runtime / comparison["runtime_seconds"]

    output_path = DATA_DIR / "tycho_real_data_method_comparison.csv"
    figure_path = FIGURE_DIR / "tycho_real_data_method_comparison.png"
    comparison.round(6).to_csv(output_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    labels = comparison["comparison_method"]
    axes[0].bar(labels, comparison["runtime_seconds"], color="#2b6cb0")
    axes[0].set_title("Runtime")
    axes[0].set_ylabel("Seconds")

    axes[1].bar(labels, comparison["predictive_count_interval_coverage_95"], color="#2f855a")
    axes[1].axhline(0.95, color="black", linewidth=1, linestyle=":")
    axes[1].set_title("Predictive coverage")
    axes[1].set_ylim(0.9, 1.0)

    axes[2].bar(labels, comparison["D_mean_vs_hsgp_vi"], color="#c05621")
    axes[2].set_title("Posterior mean distortion")
    axes[2].set_ylabel("D_mean")

    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(comparison.round(4).to_string(index=False))
    print(output_path)
    print(figure_path)


if __name__ == "__main__":
    main()
