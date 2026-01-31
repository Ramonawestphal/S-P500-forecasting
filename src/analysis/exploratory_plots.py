import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data.loader import load_spx_data


def run_exploratory_plots():
    data = load_spx_data()

    dates = data["date"]
    returns = data["returns"]
    rv5 = data["rv5"]
    vix = data["vix"]

    # ---------------------------
    # Log returns
    # ---------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(dates, returns, label="Returns")
    plt.title("S&P 500 Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Realized variance
    # ---------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(dates, rv5, label="RV5", color="black")
    plt.title("Realized Variance (RV5)")
    plt.xlabel("Date")
    plt.ylabel("RV")
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # VIX
    # ---------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(dates, vix, label="VIX", color="darkred")
    plt.title("VIX Index")
    plt.xlabel("Date")
    plt.ylabel("VIX")
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_exploratory_plots()
