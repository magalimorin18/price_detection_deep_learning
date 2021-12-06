"""Evaluate some results."""
# pylint: skip-file
from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def evaluate_results(*, predictions_path: str, truth_path: str, plot: bool = False) -> float:
    """Evaluate the results.

    The format of the data (truth) should be:
    id,img_name,x1,y1,x2,y2,price

    The format of the predictions should be:
    id,img_name,x1,y1,x2,y2,price,confidence

    AND SHOULD CONTAIN THE SAME NUMBER OF LINES
    """
    # Load the data
    pred_df = pd.read_csv(predictions_path, index_col=0)
    truth_df = pd.read_csv(truth_path, index_col=0)

    if pred_df.shape[0] != truth_df.shape[0]:
        raise ValueError("Predictions and Truth should have the same number of lines")
    pred_df["groundtruth"] = truth_df["price"]

    del truth_df

    # Evaluate the results, average precision
    pred_df = pred_df.sort_values(by="confidence", ascending=False)

    true_positive = np.array(pred_df["price"] == pred_df["groundtruth"]).astype(float)
    positive = np.ones(pred_df.shape[0])
    num_positives = pred_df.shape[0]

    true_positive_cumulative = np.cumsum(true_positive)
    positive_cumsum = np.cumsum(positive)

    precision = true_positive_cumulative / positive_cumsum
    coverage = positive_cumsum / num_positives
    average_precision = np.mean(precision)

    if plot:
        plt.plot(coverage, precision)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        plt.title("Precision of the model")

    print(f"Average precision: {average_precision}")
    return average_precision


def generate_false_results(*, truth_path: str, path: str, random_range: float = 2.0) -> False:
    """Generate false results from the truth."""
    pred_df = pd.read_csv(truth_path, index_col=0)
    pred_df["price"] = pred_df["price"].apply(lambda x: 0.0 if random() > 0.5 else x)
    pred_df["confidence"] = pred_df["price"].apply(lambda x: random())
    pred_df.to_csv(path, index=True)


if __name__ == "__main__":
    import os

    from src.config import TRAIN_IMAGES

    t_p = os.path.join(TRAIN_IMAGES, "annotations.csv")
    p_p = os.path.join(TRAIN_IMAGES, "pred_test.csv")
    if not os.path.isfile(p_p):
        generate_false_results(truth_path=t_p, path=p_p, random_range=2)

    evaluate_results(predictions_path=p_p, truth_path=t_p, plot=True)
    plt.show()
