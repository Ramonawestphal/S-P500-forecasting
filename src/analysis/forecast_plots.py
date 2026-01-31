import matplotlib.pyplot as plt
import numpy as np
import os

from src.experiments.out_of_sample import run_out_of_sample_experiment


def run_forecast_plots(horizon=1):
    results = run_out_of_sample_experiment()
    predictions = results["predictions"]
    targets = results["targets"]

    os.makedirs("results/plots/figures", exist_ok=True)

    horizons = sorted(targets.keys())

    for h in horizons:
        plt.figure(figsize=(10, 5))

        for model in predictions.keys():
            plt.plot(
                predictions[model][h],
                label=model,
                linewidth=2
            )

        plt.plot(
            targets[h],
            color="black",
            linestyle="--",
            label="Realized variance",
            linewidth=2
        )

    plt.title(f"Out-of-Sample Volatility Forecasts (h = {h})")
    plt.xlabel("Out-of-sample time index")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"results/plots/figures/volatility_predictions_h{h}.png")
    plt.close()


if __name__ == "__main__":
    run_forecast_plots(horizon=1)
