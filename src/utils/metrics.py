"""Metrics."""

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def metric_iou(
    image: np.ndarray, true_annotations: pd.DataFrame, pred_annotations: pd.DataFrame, display=False
) -> float:
    """Metric IOU (Intersection over Union).
    Probably not the best approach, but works for now.
    The problem is that we have several boxes for pred and true, not just one of each.
    """
    pred_image = np.zeros(image.shape[1:], dtype=np.int16)
    true_image = np.zeros(image.shape[1:], dtype=np.int16)
    for row in true_annotations.itertuples():
        pred_image[int(row.y1) : ceil(row.y2), int(row.x1) : ceil(row.x2)] = 1
    for row in pred_annotations.itertuples():
        true_image[int(row.y1) : ceil(row.y2), int(row.x1) : ceil(row.x2)] = 1

    score = np.sum((pred_image == true_image) & (pred_image == 1)) / (
        np.sum(pred_image + true_image > 0) + 1e-6
    )
    if display:
        fig, ax = plt.subplots()
        ax.imshow((pred_image + true_image) / 2.0)
        return score, fig
    return score
