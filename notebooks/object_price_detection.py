# +
"""Distance object price."""
# pylint: disable=wrong-import-position,invalid-name
# %load_ext autoreload
# %autoreload 2

import logging
import os
import sys
from random import choice

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread

sys.path.append(os.path.abspath(".."))
from src.config import TRAIN_ANNOTATIONS, TRAIN_IMAGES, TRAIN_PRICE_LOCATIONS
from src.display.display_image import display_annotations, display_image
from src.processing.vott import load_vott_data
from src.utils.distances import (
    compute_price_positions,
    compute_product_positions,
    find_closest_price,
)

logging.basicConfig(level=logging.INFO)
# -

products = pd.read_csv(TRAIN_ANNOTATIONS).set_index("id")
products.head()

prices = load_vott_data(TRAIN_PRICE_LOCATIONS)
prices.head()

# +
# Compute the positions of each product and each price
prices = compute_price_positions(prices)
products = compute_product_positions(products)

products.head()
# -

prices.head()

# +
image_name = choice(prices["img_name"].unique())
image_path = os.path.join(TRAIN_IMAGES, image_name)
print(f"We will display the image {image_name}")

relevant_products = products[products["img_name"] == image_name]
relevant_prices = prices[prices["img_name"] == image_name]

fig, ax = plt.subplots(1, 1, figsize=(30, 15))
display_image(imread(image_path), ax=ax)


display_annotations(annotations=relevant_products, ax=ax, color=0)
display_annotations(annotations=relevant_prices, ax=ax, color=1)
# -

product_to_price_list = []
for i, product in relevant_products.iterrows():
    product_to_price_list.append(find_closest_price(product, relevant_prices))
product_to_price = pd.DataFrame.from_records(product_to_price_list)
product_to_price.head()

# +
fig, ax = plt.subplots(1, 1, figsize=(30, 15))
display_image(imread(image_path), ax=ax)


display_annotations(annotations=relevant_products, ax=ax, color=0)
display_annotations(annotations=relevant_prices, ax=ax, color=1)

for product in product_to_price.itertuples():
    ax.plot([product.pos_x, product.price_x], [product.pos_y, product.price_y], c="r")
