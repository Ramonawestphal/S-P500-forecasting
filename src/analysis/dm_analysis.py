import numpy as np
from src.evaluation.dm_test import diebold_mariano

def run_dm_tests(oos_results):
    preds = oos_results["predictions"]
    targets = oos_results["targets"]

    dm_stat, p_val = diebold_mariano(
        loss_1=(np.array(preds["SGARCH"][1]) - np.array(targets[1]))**2,
        loss_2=(np.array(preds["ARGARCH"][1]) - np.array(targets[1]))**2,
        h=1
    )

    print(f"DM statistic: {dm_stat:.3f}")
    print(f"p-value: {p_val:.2e}")

    return {
        "dm_stat": dm_stat,
        "p_value": p_val
    }
