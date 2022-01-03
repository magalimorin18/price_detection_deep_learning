# +
"""Search for the best hyperp   arams for the object detection model."""
# pylint: disable=wrong-import-position,invalid-name,expression-not-assigned,no-member,pointless-statement
# %load_ext autoreload
# %autoreload 2

import sys

sys.path.append("..")

from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

plt.style.use("dark_background")

from src.models.utils import find_best_model, get_params_from_distributions

# -

# Define the distributions of each hyperparameter
params_config = {
    "model_type": ["resnet50", "mobilnetv3"],
    "OPTI_LEARNING_RATE": stats.loguniform(1e-5, 1e-1),
    "OPTI_BETA": (0.9, 0.999),
    "OPTI_WEIGHT_DECAY": stats.loguniform(1e-10, 1e-3),
    "epochs": stats.uniform(2, 10),
}

# +
params_to_study = [k for k, v in params_config.items() if hasattr(v, "rvs")]
print(params_to_study)
fig, axes = plt.subplots(len(params_to_study), 2, figsize=(20, len(params_to_study) * 5))

for ax, param_to_study in zip(axes, params_to_study):
    ax[0].set_title(f"{param_to_study} ()")
    sns.histplot(params_config[param_to_study].rvs(size=1000), ax=ax[0], log_scale=False)

    ax[1].set_title(f"{param_to_study} (Log)")
    sns.histplot(params_config[param_to_study].rvs(size=1000), ax=ax[1], log_scale=True)
plt.show()
# -

print("Sample of the different parameters:")
pprint(get_params_from_distributions(params_config))

# pylint: disable=using-constant-test
if True:
    find_best_model(params_config=params_config, n=20)

df = pd.read_csv("../data/training_results.csv")
df.head()

df

df.sort_values(by="loss_objectness")
