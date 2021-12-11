"""Number extraction."""
# +
# pylint: disable=wrong-import-position,invalid-name,pointless-statement
# %load_ext autoreload
# %autoreload 2

import logging
import os
import sys
from random import choice

import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(".."))
from src.data.price_locations import PriceLocationsDataset
from src.display.display_image import display_annotations, display_image
from src.models.digit_detector import DigitDetector
from src.processing.croping import crop_image

logging.basicConfig(level=logging.INFO)
# -

# Load the dataset
dataset = PriceLocationsDataset()

# +
img_name = os.path.basename(choice(dataset.unique_images))
print(f"We select the following image for the process: {img_name}")

img = dataset.get_original_image(img_name)
annotations = dataset.get_all_annotations_for_one_image(img_name)

fig, ax = plt.subplots(1, 1, figsize=(30, 15))

display_image(img, ax=ax)
display_annotations(annotations, ax=ax)
# -

price_imgs = []
for annotation in annotations.itertuples():
    price_imgs.append(
        crop_image(
            img, int(annotation.x1), int(annotation.y1), int(annotation.x2), int(annotation.y2)
        )
    )
print(len(price_imgs))

# +
W = 4
H = 5
fig, axes = plt.subplots(W, H, figsize=(15, 10))

for w in range(W):
    for h in range(H):
        i = w * H + h
        price_img = price_imgs[i]
        ax = axes[w][h]
        ax.imshow(price_img)
# -
img = price_imgs[3]
plt.imshow(img)
img.shape


cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

digit_detector = DigitDetector()

digits_locations = (
    digit_detector.extract_digits_locations(img).reset_index().rename(columns={"index": "price"})
)
digits_locations.head()

# +
fig, ax = plt.subplots(1, 1)

display_image(img, ax=ax)
display_annotations(digits_locations, ax=ax)
# -

loc = digits_locations.iloc[7]
loc

digit = digit_detector.prepare_img_for_detection(img, loc.x1, loc.y1, loc.x2, loc.y2)
plt.imshow(digit, cmap="gray")
