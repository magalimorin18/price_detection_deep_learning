"""Number extraction."""
# +
# pylint: disable=wrong-import-position,invalid-name,pointless-statement,redefined-outer-name,expression-not-assigned
# %load_ext autoreload
# %autoreload 2

import logging
import math
import os
import sys
from random import choice

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering

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

complete_image = dataset.get_original_image(img_name)
annotations = dataset.get_all_annotations_for_one_image(img_name)

fig, ax = plt.subplots(1, 1, figsize=(30, 15))

display_image(complete_image, ax=ax)
display_annotations(annotations, ax=ax)
# -

price_imgs = []
for annotation in annotations.itertuples():
    price_imgs.append(
        crop_image(
            complete_image,
            int(annotation.x1),
            int(annotation.y1),
            int(annotation.x2),
            int(annotation.y2),
        )
    )
print(f"We have found {len(price_imgs)} price tags")

# +
W = 4
H = max(2, math.ceil(len(price_imgs) / W - 1e-3))
fig, axes = plt.subplots(W, H, figsize=(15, 10))

for w in range(W):
    for h in range(H):
        i = w * H + h
        if i < len(price_imgs):
            price_img = price_imgs[i]
            ax = axes[w][h]
            ax.imshow(price_img)
# +
def transform_image(img):
    """Function to transform an image."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = (img>180).astype(float)
    img = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 3)
    return img


W = 4
H = math.ceil(len(price_imgs) / W - 1e-3)
fig, axes = plt.subplots(W, H, figsize=(15, 10))

for w in range(W):
    for h in range(H):
        i = w * H + h
        if i < len(price_imgs):
            price_img = price_imgs[i]
            ax = axes[w][h]
            ax.imshow(transform_image(price_img), cmap="gray")
# -

img = price_imgs[11]
plt.imshow(img)
img.shape


# +
grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_image = cv2.GaussianBlur(grey_image, (5, 5), 0)
thresh_image = cv2.adaptiveThreshold(blur_image, 255, 1, 1, 11, 3)
contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

plt.imshow(thresh_image, cmap="gray")
# -

test_img = img.copy()
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
test_img = (test_img > 140).astype(float)
plt.imshow(test_img, cmap="gray")

digit_detector = DigitDetector()

digits_locations = digit_detector.extract_digits_locations(img)
digits_locations.head()

# +
digits_locations["price"] = digits_locations["result"].apply(lambda x: x[0])
digits_locations["proba"] = digits_locations["result"].apply(lambda x: x[1] * 100)
digits_locations["price"] = digits_locations.index

fig, ax = plt.subplots(1, 1)
display_image(img, ax=ax)
display_annotations(digits_locations, ax=ax)

# +
digits_locations["w"] = digits_locations.apply(lambda row: row.x2 - row.x1, axis=1)
digits_locations["h"] = digits_locations.apply(lambda row: row.y2 - row.y1, axis=1)
digits_locations["wh_ratio"] = digits_locations.apply(lambda row: row.h / row.w, axis=1)
digits_locations["area"] = digits_locations.apply(lambda row: row.w * row.h, axis=1)
digits_locations["x"] = digits_locations.apply(lambda row: (row.x1 + row.x2) / 2, axis=1)
digits_locations["y"] = digits_locations.apply(lambda row: (row.y1 + row.y2) / 2, axis=1)

digits_locations.head()

# +
plt.style.use("dark_background")
df = digits_locations.copy()
df = df[df["wh_ratio"] > 1.15]

plt.scatter(df["wh_ratio"], df["h"], s=4, marker="+")

# +
digits_locations = digits_locations[digits_locations["wh_ratio"] > 1.15]

digits_locations["y"].sort_values().plot.bar()

# +

clustering_model = AgglomerativeClustering(n_clusters=2)
print(digits_locations.index)
clustering_model.fit(digits_locations[["x", "y", "w", "h"]])
print(clustering_model.labels_)
print(clustering_model.children_)

final_digits = []
max_length = digits_locations.shape[0]
for a, b in clustering_model.children_:
    if a < max_length and b < max_length:
        final_digits.extend([a, b])
    if len(final_digits) >= 6:
        break
final_digits_indexes = [digits_locations.index[x] for x in final_digits]
print(final_digits)
print("final digits (index)", final_digits_indexes)
# -

fig, ax = plt.subplots(1, 1)
display_image(img, ax=ax)
display_annotations(digits_locations.iloc[final_digits], ax=ax)

loc = digits_locations.iloc[final_digits_indexes[0]]
print(final_digits_indexes[0], loc)
digit = digit_detector.prepare_img_for_detection(
    complete_image, int(loc.x1), int(loc.y1), int(loc.x2), int(loc.y2)
)
plt.imshow(np.array(digit[0]), cmap="gray")

complete_image[int(loc.y1) : int(loc.y2), int(loc.x1) : int(loc.x2)].shape
