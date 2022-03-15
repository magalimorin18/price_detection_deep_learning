"""Annotate digit"""
# #!jupyter nbextension enable --py jupyter_bbox_widget
# pylint: disable=wrong-import-position,invalid-name,pointless-statement
# %load_ext autoreload
# %autoreload 2

# +
import logging
import os
import sys

import pandas as pd
from jupyter_bbox_widget import BBoxWidget

sys.path.append(os.path.abspath(".."))
from PIL import Image

from src.config import (
    TEST_DIGIT_LOCATIONS_CSV_FOLDER,
    TEST_DIGIT_LOCATIONS_IMAGES_FOLDER,
    TRAIN_DIGIT_LOCATIONS_CSV_FOLDER,
    TRAIN_DIGIT_LOCATIONS_IMAGES_FOLDER,
    VAL_DIGIT_LOCATIONS_CSV_FOLDER,
    VAL_DIGIT_LOCATIONS_IMAGES_FOLDER,
)
from src.data.price_locations import PriceLocationsDataset
from src.processing.annotations import (
    load_annotations_one_price_tag,
    save_annotations_one_price_tag,
)

LABEL = list(map(str, range(10)))

logging.basicConfig(level=logging.INFO)
# -

dataset = PriceLocationsDataset()

df = pd.read_csv("../data/train/price_tags.csv")
df.head()


def get_price_tag_image(price_tag_id):
    """
    Returns the price tag image
    """
    price_tag = df[df["index"] == price_tag_id]
    img_name = price_tag["img_name"].iloc[0]
    image = dataset.get_original_image(img_name)
    x1 = price_tag["x1"]
    y1 = price_tag["y1"]
    x2 = price_tag["x2"]
    y2 = price_tag["y2"]
    cropped_img = image[int(y1) : int(y2), int(x1) : int(x2)]
    return cropped_img


def save_image(price_tag_id):
    """
    Save the image
    """
    imgpil = Image.fromarray(
        get_price_tag_image(price_tag_id)
    )  # Transformation du tableau en image PIL
    imgpil.save(f"../data/train/digit_tags/{str(price_tag_id).rjust(5, '0')}.jpg")


# +
price_tag_index = "00099.jpg"
dataset_type = "val"

index = 0
if dataset_type == "train":
    image_path = os.path.join(TRAIN_DIGIT_LOCATIONS_IMAGES_FOLDER, price_tag_index)
    csv_path = os.path.join(TRAIN_DIGIT_LOCATIONS_CSV_FOLDER, price_tag_index)
elif dataset_type == "test":
    image_path = os.path.join(TEST_DIGIT_LOCATIONS_IMAGES_FOLDER, price_tag_index)
    csv_path = os.path.join(TEST_DIGIT_LOCATIONS_CSV_FOLDER, price_tag_index)
elif dataset_type == "val":
    image_path = os.path.join(VAL_DIGIT_LOCATIONS_IMAGES_FOLDER, price_tag_index)
    csv_path = os.path.join(VAL_DIGIT_LOCATIONS_CSV_FOLDER, price_tag_index)

print(image_path)

bboxes_df = load_annotations_one_price_tag(image_path)
bboxes = [
    {"x": row.x1, "y": row.y1, "width": row.x2 - row.x1, "height": row.y2 - row.y1, "tag": LABEL}
    for row in bboxes_df.itertuples()
]
print(f"Found {len(bboxes)} boxes")
# -

widget = BBoxWidget(
    image=".." + image_path.split("..")[-1].replace("\\", "/"), classes=LABEL, bboxes=bboxes
)
widget

df = pd.DataFrame.from_records(widget.bboxes)
df = df.rename(columns={"x": "x1", "y": "y1"})
df["x2"] = df["x1"] + df["width"]
df["y2"] = df["y1"] + df["height"]
df["img_tags"] = price_tag_index
df = df.drop(columns=["width", "height", "tag"], errors="ignore")
save_annotations_one_price_tag(price_tag_index, df, dataset=dataset_type)
