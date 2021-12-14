"""Annotations functions."""

import logging
import os

import pandas as pd

from src.config import TRAIN_PRICE_LOCATIONS_FILE, TRAIN_PRICE_LOCATIONS_FOLDER


def load_annotations_one_image(image_name: str) -> pd.DataFrame:
    """Load all the annotations."""
    path = os.path.join(TRAIN_PRICE_LOCATIONS_FOLDER, os.path.basename(image_name)).replace(
        ".jpg", ".csv"
    )
    if os.path.isfile(path):
        return pd.read_csv(path)
    logging.warning("No annotations found for %s", image_name)
    return pd.DataFrame()


def save_annotations_one_image(image_name: str, annotations: pd.DataFrame) -> None:
    """Save all the annotations."""
    path = os.path.join(TRAIN_PRICE_LOCATIONS_FOLDER, os.path.basename(image_name))
    print(path)
    annotations.to_csv(path.replace(".jpg", ".csv"), index=False)


def combine_all_annotations() -> None:
    """Combine all the annotations."""
    dfs = []
    for filename in os.listdir(TRAIN_PRICE_LOCATIONS_FOLDER):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(TRAIN_PRICE_LOCATIONS_FOLDER, filename))
            df["img_name"] = filename.replace(".csv", ".jpg")
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(TRAIN_PRICE_LOCATIONS_FILE), index=False)


def load_price_annotations() -> pd.DataFrame:
    """Load the price annotations."""
    if os.path.isfile(TRAIN_PRICE_LOCATIONS_FILE):
        annotations: pd.DataFrame = pd.read_csv(TRAIN_PRICE_LOCATIONS_FILE)
        return annotations
    raise FileNotFoundError(f"No annotations found for {TRAIN_PRICE_LOCATIONS_FILE}")
