"""Configuration file."""

import os

# Classic locations
SRC_FOLDER = os.path.abspath(os.path.dirname(__file__))
ROOT_FOLDER = os.path.join(SRC_FOLDER, "..")
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")


# Dataset specifics
TEST_ANNOTATIONS = os.path.join(DATA_FOLDER, "test", "results.csv")
TEST_IMAGES = os.path.join(DATA_FOLDER, "test", "images")

TRAIN_ANNOTATIONS = os.path.join(DATA_FOLDER, "train", "results.csv")
TRAIN_IMAGES = os.path.join(DATA_FOLDER, "train", "images")

TRAIN_PRICE_LOCATIONS = os.path.join(DATA_FOLDER, "vott-csv-export", "Price-detection-export.csv")
