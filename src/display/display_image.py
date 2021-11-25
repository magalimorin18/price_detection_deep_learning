"""Display one image."""
from typing import Optional

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


def display_image(
    img: np.ndarray,
    annotations: Optional[pd.DataFrame] = None,
    ax: Optional[plt.Axes] = None,
    color: str = "#00ff08",
) -> None:
    """Display one image."""
    if ax is None:
        ax = plt.gca()
    ax.imshow(img)
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
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
        txt = ax.text(annotation.x1, annotation.y1, annotation.price, color=color)
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=5, foreground="#000000F0")]
        )
