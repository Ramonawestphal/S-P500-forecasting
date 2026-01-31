from src.experiments.out_of_sample import run_out_of_sample_experiment
from src.evaluation.losses import qlike_loss
from src.evaluation.dm_test import diebold_mariano
import numpy as np

results = run_out_of_sample_experiment()

predictions = results["predictions"]
targets = results["targets"]

# Example: 1-day ahead
pred_sgarch = np.array(results["predictions"]["SGARCH"][1])
pred_argarch = np.array(results["predictions"]["ARGARCH"][1])
target = np.array(results["targets"][1])


loss_sgarch = qlike_loss(pred_sgarch, target)
loss_argarch = qlike_loss(pred_argarch, target)

dm_stat, p_val = diebold_mariano(
    loss_1=(pred_sgarch - target) ** 2,
    loss_2=(pred_argarch - target) ** 2,
    h=1
)

print("DM statistic:", dm_stat)
print("p-value:", p_val)
