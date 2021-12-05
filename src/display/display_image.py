"""Display one image."""

from typing import List, Optional, Union

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

COLOR_LIST: List[str] = ["#00ff08", "#ff0f0f", "#ec008c", "#4DC6E2"]


def display_image(img: np.ndarray, ax: Optional[plt.Axes] = None) -> None:
    """Display one image."""
    if ax is None:
        ax = plt.gca()

    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    ax.imshow(img)
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)


def display_annotations(
    annotations: Optional[pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    color: Union[int, str] = 0,
    display_price: bool = True,
):
    """Display annotations.

    The annotations dataframe should have the following columns:
    x1,x2,y1,y2,label,price
    If price is not present, it will not be displayed.
    """
    if isinstance(color, int):
        color = COLOR_LIST[color]
    if "price" not in annotations:
        display_price = False
    for annotation in annotations.itertuples():
        ax.add_patch(
            patches.Rectangle(
                (annotation.x1, annotation.y1),
                annotation.x2 - annotation.x1,
                annotation.y2 - annotation.y1,
                color=color,
                fill=None,
            )
        )
        if display_price:
            txt = ax.text(annotation.x1, annotation.y1, annotation.price, color=color)
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground="#000000F0")])
