# +
# Setup the notebook
import os
import sys

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(".."))
np.random.seed(666)

# %load_ext autoreload
# %autoreload 2
# -

from src.config import Config
from src.display.display_image import display_image
from src.processing.loading import AnnotationHandler, ImageDataset

# ## Playing with the dataset

# +
annotations = AnnotationHandler()

annotations[1].head()
# -

dataset = ImageDataset()

dataset[0].shape

# ## Display some images with annotations

W, H = 2, 2
fig, axes = plt.subplots(W, H, figsize=(20, 10))
for i in range(W):
    for j in range(H):
        idx = i * H + j
        ax = axes[i, j]
        display_image(dataset[idx], annotations[idx], ax)
plt.show()
