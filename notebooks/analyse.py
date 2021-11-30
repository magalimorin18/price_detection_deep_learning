"""Analysis of the data."""
# +
# %load_ext autoreload
# %autoreload 2

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches
from PIL import Image

sys.path.append(os.path.abspath("..."))
# -

# pylint: disable=unsubscriptable-object
annotations: pd.DataFrame = pd.read_csv("../data/annotations.csv")

annotations.head()


# +
# from collections import OrderedDict  (je sais pas si ca va me servir)
# noms_images = []
# for i in annotations.index:
# noms_images.append(annotations.iloc[i,1])
# final_noms_images = list(OrderedDict.fromkeys(noms_images))


# pylint: disable=too-many-locals
def display_image_with_price(
    image_name: str,
):
    """Display the image with boxes and prices.

    image_name is a string with the following shape: "0001.jpg"
    """
    x1_list = []
    x2_list = []
    y1_list = []
    y2_list = []
    price_list = []
    rectangles = []

    for annotation in annotations[annotations["img_name"] == image_name].itertuples():
        x1_list.append(annotation.x1)
        y1_list.append(annotation.y1)
        x2_list.append(annotation.x2)
        y2_list.append(annotation.y2)
        price_list.append(annotation.price)

    for x1, x2, y1, y2, price in zip(x1_list, x2_list, y1_list, y2_list, price_list):
        rect = [
            patches.Rectangle(
                (x1, y1),
                (x2 - x1),
                (y2 - y1),
                linewidth=2,
                edgecolor="cyan",
                fill=False,
            ),
            x1,
            y2,
            str(price),
        ]

        rectangles.append(rect)

    path = os.path.join("..", "data", "images", image_name)
    im = Image.open(path)
    plt.imshow(im)
    ax = plt.gca()
    for rectangle in rectangles:
        ax.add_patch(rectangle[0])
        plt.text(
            rectangle[1],
            rectangle[2],
            rectangle[3],
            backgroundcolor="r",
            color="b",
            fontsize="x-small",
            fontweight="bold",
        )
    plt.show()


# -

for i in range(0, 6):
    display_image_with_price("000" + str(i) + ".jpg")
