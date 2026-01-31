"""
Main experiment runner for S&P 500 volatility forecasting
"""

from src.experiments.in_sample import run_in_sample_experiment
from src.experiments.out_of_sample import run_out_of_sample_experiment
from src.analysis.dm_analysis import run_dm_tests
from src.analysis.forecast_plots import run_forecast_plots

def main():
    print("=" * 60)
    print("S&P 500 Volatility Forecasting – Full Experiment")
    print("=" * 60)

    # ------------------
    # 1. In-sample
    # ------------------
    print("\n[1] Running in-sample estimation...")
    in_sample_results = run_in_sample_experiment()

    # ------------------
    # 2. Out-of-sample
    # ------------------
    print("\n[2] Running out-of-sample forecasts...")
    oos_results = run_out_of_sample_experiment()

    # ------------------
    # 3. DM tests
    # ------------------
    print("\n[3] Running Diebold–Mariano tests...")
    dm_results = run_dm_tests(oos_results)

    # ------------------
    # 4. Plots
    # ------------------
    print("\n[4] Generating plots...")
    run_forecast_plots(oos_results)

    print("\n✅ All experiments completed successfully.")

if __name__ == "__main__":
    main()
