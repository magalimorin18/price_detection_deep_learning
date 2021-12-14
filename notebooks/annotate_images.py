# %%
"""Annotate images"""
# !pip install jupyter_bbox_widget --quiet
# pylint: disable=wrong-import-position,invalid-name,pointless-statement
# %load_ext autoreload
# %autoreload 2

# %%
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from jupyter_bbox_widget import BBoxWidget

sys.path.append(os.path.abspath(".."))
from src.config import TRAIN_IMAGES, TRAIN_PRICE_LOCATIONS_FOLDER
from src.models.price_detector import PriceDetector
from src.processing.annotations import (
    combine_all_annotations,
    load_annotations_one_image,
    save_annotations_one_image,
)

LABEL = "price_tag"

logging.basicConfig(level=logging.INFO)

# %%
image_name = "0140.jpg"
image_path = os.path.join(TRAIN_IMAGES, image_name)
print(image_path)

bboxes_df = load_annotations_one_image(image_path)
bboxes = [
    {"x": row.x1, "y": row.y1, "width": row.x2 - row.x1, "height": row.y2 - row.y1, "tag": LABEL}
    for row in bboxes_df.itertuples()
]
print(f"Found {len(bboxes)} boxes")


# %%
widget = BBoxWidget(
    image=".." + image_path.split("..")[-1].replace("\\", "/"), classes=[LABEL], bboxes=bboxes
)
widget

# %%
df = pd.DataFrame.from_records(widget.bboxes)
print(df.head())
df = df.rename(columns={"x": "x1", "y": "y1"})
df["x2"] = df["x1"] + df["width"]
df["y2"] = df["y1"] + df["height"]
df = df.drop(columns=["width", "height", "tag", "label"], errors="ignore")

save_annotations_one_image(image_name, df)

# %%
combine_all_annotations()

# %%
# Predict with the model on new images
model = PriceDetector()

# %%
image_paths = [f"../data/train/images/{str(index).rjust(4, '0')}.jpg" for index in range(110, 150)]
images = [plt.imread(path) for path in image_paths]
print(len(images))

# %%
model_output = model.extract_prices_locations(images)

# %%
for image_path, pred in zip(image_paths, model_output):
    print(pred)
    pred = pred.drop(columns=["score", "price"])
    pred.to_csv(
        os.path.join(
            TRAIN_PRICE_LOCATIONS_FOLDER, os.path.basename(image_path.replace(".jpg", ".csv"))
        )
    )


# %%
