"""Metrics."""

from math import ceil

import numpy as np
import pandas as pd


def metric_iou(
    image: np.ndarray, true_annotations: pd.DataFrame, pred_annotations: pd.DataFrame
) -> float:
    """Metric IOU (Intersection over Union).
    Probably not the best approach, but works for now.
    The problem is that we have several boxes for pred and true, not just one of each.
    """
    masked_image = np.zeros(image.shape[:2])
    for row in true_annotations.itertuples():
        masked_image[int(row.y1) : ceil(row.y2), int(row.x1) : ceil(row.x2)] = 1
    for row in pred_annotations.itertuples():
        masked_image[int(row.y1) : ceil(row.y2), int(row.x1) : ceil(row.x2)] += 1
    return np.sum(masked_image == 2) / (np.sum(masked_image > 0) + 1e-6)
