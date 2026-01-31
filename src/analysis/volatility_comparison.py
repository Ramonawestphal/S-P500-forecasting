import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from src.data.loader import load_spx_data
from src.models.garch import estimate_sgarch
from src.models.agarch import estimate_agarch
from src.models.realgarch import (
    estimate_realgarch_symmetric,
    estimate_realgarch_asymmetric
)


def run_volatility_comparison():
    data = load_spx_data()

    dates = data["date"]
    returns = data["returns"].values
    rv5 = data["rv5"].values

    # ---------------------------
    # Estimate models
    # ---------------------------
    params_sg, sigma2_sg, _ = estimate_sgarch(returns)
    params_ag, sigma2_ag, _ = estimate_agarch(returns)

    params_rs, sigma2_rs, _ = estimate_realgarch_symmetric(returns, rv5)
    params_ra, sigma2_ra, _ = estimate_realgarch_asymmetric(returns, rv5)

    # ---------------------------
    # Plot
    # ---------------------------
    plt.figure(figsize=(13, 6))
    plt.plot(dates, np.sqrt(sigma2_sg), label="SGARCH")
    plt.plot(dates, np.sqrt(sigma2_ag), label="AGARCH", linestyle="--")
    plt.plot(dates, np.sqrt(sigma2_rs), label="SRGARCH")
    plt.plot(dates, np.sqrt(sigma2_ra), label="ARGARCH", linestyle="--")

    plt.title("Filtered Volatility Comparison")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_volatility_comparison()
