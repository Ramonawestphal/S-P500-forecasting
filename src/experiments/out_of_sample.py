import numpy as np

from src.models.realgarch_x import estimate_realgarch_x_2step
from src.data.loader import load_spx_data
from src.models.garch import estimate_garch
from src.models.agarch import estimate_agarch
from src.models.realgarch import estimate_realgarch_symmetric_2step, estimate_realgarch_asymmetric_2step
from src.forecasting.garch_forecasts import (forecast_garch, forecast_agarch)
from src.forecasting.realgarch_forecasts import (forecast_realgarch_symmetric, forecast_realgarch_asymmetric)
from src.models.utils import extract_params

def run_out_of_sample_experiment(
    split_ratio=0.8,
    horizons=(1, 5, 21)
):
    # ---------------------
    # Load data
    # ---------------------
    data = load_spx_data()
    returns = data["returns"]
    rv5 = data["rv5"]

    T = len(returns)
    split = int(T * split_ratio)

    returns_in, returns_out = returns[:split], returns[split:]
    rv5_in, rv5_out = rv5[:split], rv5[split:]

    # ---------------------
    # Estimate models (in-sample)
    # ---------------------
    params_garch = extract_params(estimate_garch(returns_in))
    params_agarch = extract_params(estimate_agarch(returns_in))
    
    params_rsym = extract_params(estimate_realgarch_symmetric_2step(returns_in, rv5_in))
    params_rasym = extract_params(estimate_realgarch_asymmetric_2step(returns_in, rv5_in))
    
    # ---------------------
    # Storage
    # ---------------------
    predictions = {m: {h: [] for h in horizons}
                   for m in ["SGARCH", "AGARCH", "SRGARCH", "ARGARCH"]}

    targets = {h: [] for h in horizons}

    # ---------------------
    # Rolling forecasts
    # ---------------------
    for t in range(len(returns_out) - max(horizons)):
        r_hist = np.concatenate([returns_in, returns_out[:t+1]])
        rv_hist = np.concatenate([rv5_in, rv5_out[:t+1]])

        for h in horizons:
            predictions["SGARCH"][h].append(
                forecast_garch(params_garch, r_hist, h)
            )

            predictions["AGARCH"][h].append(
                forecast_agarch(params_agarch, r_hist, h)   # ✅ FIX
            )

            predictions["SRGARCH"][h].append(
                forecast_realgarch_symmetric(params_rsym, r_hist, rv_hist, h)
            )

            predictions["ARGARCH"][h].append(
                forecast_realgarch_asymmetric(params_rasym, r_hist, rv_hist, h)
            )

            targets[h].append(np.sum(rv5_out[t+1:t+h+1]))


    return {
        "predictions": predictions,
        "targets": targets
    }
