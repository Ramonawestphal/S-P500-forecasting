S&P 500 Volatility Forecasting with GARCH Models

This project implements and compares multiple volatility forecasting models for the S&P 500, with a focus on out-of-sample predictive performance.
The framework is fully modular, reproducible, and runnable with a single command.

**Project Overview**

We forecast S&P 500 volatility using:

- GARCH(1,1) (symmetric)

- Asymmetric GARCH (AGARCH)

- Realized GARCH (symmetric & asymmetric)

- HAR-RV

- VIX-based forecasts

Models are evaluated out-of-sample using:

- Mean Squared Error (MSE)

- QLIKE loss

- Diebold–Mariano tests for predictive accuracy comparison

The project was developed as part of a quantitative finance / financial econometrics case study.

**Repository Structure**
.
├── README.md
├── requirements.txt
│
├── data/                     # Raw and processed input data
│   ├── sp500.csv
│   ├── rv5_*.csv
│   └── indeces/
│
├── results/
│   └── plots/
│       └── figures/
│           ├── volatility_predictions_h1.png
│           ├── volatility_predictions_h5.png
│           └── volatility_predictions_h21.png
│
└── src/
    ├── main.py               # One-click experiment runner
    │
    ├── data/
    │   └── loader.py          # Data loading & preprocessing
    │
    ├── models/
    │   ├── garch.py
    │   ├── agarch.py
    │   ├── realgarch.py
    │   ├── realgarch_x.py
    │   ├── har_rv.py
    │   ├── filters.py
    │   ├── likelihoods.py
    │   └── utils.py
    │
    ├── forecasting/
    │   ├── garch_forecasts.py
    │   ├── realgarch_forecasts.py
    │   ├── har_forecasts.py
    │   └── vix_forecast.py
    │
    ├── experiments/
    │   ├── in_sample.py
    │   └── out_of_sample.py
    │
    └── analysis/
        ├── exploratory_plots.py
        ├── volatility_comparison.py
        ├── forecast_plots.py
        └── out_of_sample_analysis.py

**How to Run**
1. Install dependencies
pip install -r requirements.txt

2. Run the full experiment (recommended)
python -m src.main


This will automatically:

- Estimate all models in-sample

- Generate rolling out-of-sample forecasts

- Compute MSE, QLIKE, and Diebold–Mariano tests

- Save comparison plots to results/plots/figures/

**Output**
Forecast plots

Saved to:

``results/plots/figures/´´


Example:

- volatility_predictions_h1.png

- volatility_predictions_h5.png

- volatility_predictions_h21.png

Statistical evaluation:

Printed directly to the console, including: 

- MSE & QLIKE loss tables

- Diebold–Mariano test statistics and p-values

**Methodological Notes**

Realized GARCH models are estimated using two-step estimation

Forecasts are strictly out-of-sample

Loss functions follow standard volatility forecasting literature

Code is written for clarity, extensibility, and reproducibility

**Technologies Used**

Python (NumPy, SciPy, pandas, matplotlib)

Optimization via scipy.optimize

Econometric modeling & forecasting

Modular research-grade code structure

**Possible Extensions**

- Multivariate GARCH

- Alternative realized volatility measures

- Bayesian estimation

- High-frequency intraday forecasting

- Portfolio-level risk applications

**Author**

Developed as part of a quantitative finance / econometrics project.
Feel free to reach out for questions or collaboration.