# +
# %load_ext autoreload
# %autoreload 2

import os
import sys

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import image
from PIL import Image

sys.path.append(os.path.abspath("..."))
# -

annotations: pd.DataFrame = pd.read_csv("../data/annotations.csv")

annotations.head()


# +
# from collections import OrderedDict  (je sais pas si ca va me servir)
# noms_images = []
# for i in annotations.index:
# noms_images.append(annotations.iloc[i,1])
# final_noms_images = list(OrderedDict.fromkeys(noms_images))


def afficher_image_box_prix(
    image_name,
):  # image_name est un string de la forme "0001.jpg"
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    price = []
    rectangle = []

    for annotation in annotations[annotations["img_name"] == image_name].itertuples():
        x1.append(annotation.x1)
        y1.append(annotation.y1)
        x2.append(annotation.x2)
        y2.append(annotation.y2)
        price.append(annotation.price)

    for j in range(len(x1)):
        rect = [
            patches.Rectangle(
                (x1[j], y1[j]),
                (x2[j] - x1[j]),
                (y2[j] - y1[j]),
                linewidth=2,
                edgecolor="cyan",
                fill=False,
            ),
            x1[j],
            y2[j],
            str(price[j]),
        ]

        rectangle.append(rect)

    path = os.path.join("..", "data", "images", image_name)
    im = Image.open(path)
    plt.imshow(im)
    ax = plt.gca()
    for k in range(len(rectangle)):
        ax.add_patch(rectangle[k][0])
        plt.text(
            rectangle[k][1],
            rectangle[k][2],
            rectangle[k][3],
            backgroundcolor="r",
            color="b",
            fontsize="x-small",
            fontweight="bold",
        )
    plt.show()


# -

for i in range(0, 6):
    afficher_image_box_prix("000" + str(i) + ".jpg")
