# +
# Setup the notebook
import os
import sys

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.append(os.path.abspath(".."))
np.random.seed(666)

# %load_ext autoreload
# %autoreload 2
# -

from src.config import Config
from src.display.display_image import display_image
from src.processing.loading import AnnotationHandler, ImageDataset

print(f"CUDA available: {torch.cuda.is_available()}")

# ## Playing with the dataset

# +
annotations = AnnotationHandler()

annotations[1].head()
# -

dataset = ImageDataset()

dataset[0].shape

# ## Display some images with annotations

# +
W, H = 2, 2
idx_offset = 10

fig, axes = plt.subplots(W, H, figsize=(20, 10))
for i in range(W):
    for j in range(H):
        idx = i * H + j
        ax = axes[i, j]
        display_image(dataset[idx + idx_offset], annotations[idx + idx_offset], ax)
plt.show()
# -

# ## Test Yolo on it

yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# +
W, H = 2, 2

imgs = [dataset.get_path(i * H + j) for i in range(W) for j in range(H)]
# -

results = yolo_model(imgs)

results.print()

fig, axes = plt.subplots(W, H, figsize=(20, 10))
for i in range(W):
    for j in range(H):
        idx = i * H + j
        ax = axes[i, j]
        annot = (
            results.pandas()
            .xyxy[idx]
            .rename(
                columns={
                    "xmin": "x1",
                    "xmax": "x2",
                    "ymin": "y1",
                    "ymax": "y2",
                    "name": "price",
                }
            )
            .drop(columns=["confidence", "class"])
        )
        display_image(dataset[idx], annot, ax)
plt.show()
