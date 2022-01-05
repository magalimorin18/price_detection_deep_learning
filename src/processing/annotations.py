"""Annotations functions."""

import logging
import os
from typing import Literal

import pandas as pd

from src.config import (
    TEST_PRICE_LOCATIONS_FILE,
    TEST_PRICE_LOCATIONS_FOLDER,
    TRAIN_PRICE_LOCATIONS_FILE,
    TRAIN_PRICE_LOCATIONS_FOLDER,
)


def load_annotations_one_image(
    image_name: str, dataset: Literal["train", "test"] = "train"
) -> pd.DataFrame:
    """Load all the annotations."""
    path = os.path.join(
        TRAIN_PRICE_LOCATIONS_FOLDER if dataset == "train" else TEST_PRICE_LOCATIONS_FOLDER,
        os.path.basename(image_name),
    ).replace(".jpg", ".csv")
    if os.path.isfile(path):
        return pd.read_csv(path)
    logging.warning("No annotations found for %s", image_name)
    return pd.DataFrame()


def save_annotations_one_image(
    image_name: str, annotations: pd.DataFrame, dataset: Literal["train", "test"] = "train"
) -> None:
    """Save all the annotations."""
    path = os.path.join(
        TRAIN_PRICE_LOCATIONS_FOLDER if dataset == "train" else TEST_PRICE_LOCATIONS_FOLDER,
        os.path.basename(image_name),
    )
    print(path)
    annotations.to_csv(path.replace(".jpg", ".csv"), index=False)


def combine_all_annotations(dataset: Literal["train", "test"] = "train") -> None:
    """Combine all the annotations."""
    dfs = []
    folder = TRAIN_PRICE_LOCATIONS_FOLDER if dataset == "train" else TEST_PRICE_LOCATIONS_FOLDER
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, filename))
            df["img_name"] = filename.replace(".csv", ".jpg")
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(
        os.path.join(
            TRAIN_PRICE_LOCATIONS_FILE if dataset == "train" else TEST_PRICE_LOCATIONS_FILE
        ),
        index=False,
    )


def load_price_annotations(dataset: Literal["train", "test"] = "train") -> pd.DataFrame:
    """Load the price annotations."""
    file_path = TRAIN_PRICE_LOCATIONS_FILE if dataset == "train" else TEST_PRICE_LOCATIONS_FILE
    if os.path.isfile(file_path):
        annotations: pd.DataFrame = pd.read_csv(file_path)
        return annotations
    raise FileNotFoundError(f"No annotations found for {file_path}")
