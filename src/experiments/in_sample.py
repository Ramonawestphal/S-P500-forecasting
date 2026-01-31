import pandas as pd
from src.data.loader import load_spx_data
from src.models.garch import estimate_garch
from src.models.agarch import estimate_agarch
from src.models.realgarch import (
    estimate_realgarch_symmetric_2step,
    estimate_realgarch_asymmetric_2step
)


def run_in_sample_experiment():
    data = load_spx_data()

    returns = data["returns"]
    rv5 = data["rv5"]

    results = []

    # --- GARCH ---
    params, sigma2, loglik = estimate_garch(returns)
    results.append({
        "Model": "GARCH",
        "LogLikelihood": loglik
    })

    # --- AGARCH ---
    params, sigma2, loglik = estimate_agarch(returns)
    results.append({
        "Model": "AGARCH",
        "LogLikelihood": loglik
    })

    # --- Realized GARCH (Symmetric) ---
    params, sigma2, loglik = estimate_realgarch_symmetric_2step(returns, rv5)
    results.append({
        "Model": "RealGARCH-S",
        "LogLikelihood": loglik
    })

    # --- Realized GARCH (Asymmetric) ---
    params, sigma2, loglik = estimate_realgarch_asymmetric_2step(returns, rv5)
    results.append({
        "Model": "RealGARCH-A",
        "LogLikelihood": loglik
    })

    return pd.DataFrame(results).sort_values("LogLikelihood", ascending=False)


if __name__ == "__main__":
    df = run_in_sample_experiment()
    print(df)
    df.to_csv("results/plots/tables/in_sample_loglikelihoods.csv", index=False)
