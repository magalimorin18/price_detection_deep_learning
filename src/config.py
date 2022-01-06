"""Configuration file."""

import os

# Classic locations
SRC_FOLDER = os.path.abspath(os.path.dirname(__file__))
ROOT_FOLDER = os.path.join(SRC_FOLDER, "..")
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
SAVED_MODELS = os.path.join(ROOT_FOLDER, "models")


# Dataset specifics
TEST_ANNOTATIONS = os.path.join(DATA_FOLDER, "test", "results.csv")
TEST_IMAGES = os.path.join(DATA_FOLDER, "test", "images")

TRAIN_ANNOTATIONS = os.path.join(DATA_FOLDER, "train", "annotations.csv")
TRAIN_IMAGES = os.path.join(DATA_FOLDER, "train", "images")

TRAIN_PRICE_LOCATIONS_FILE = os.path.join(DATA_FOLDER, "train", "price_tags.csv")
TRAIN_PRICE_LOCATIONS_FOLDER = os.path.join(DATA_FOLDER, "train", "price_tags")
TEST_PRICE_LOCATIONS_FILE = os.path.join(DATA_FOLDER, "test", "price_tags.csv")
TEST_PRICE_LOCATIONS_FOLDER = os.path.join(DATA_FOLDER, "test", "price_tags")


# MODELS
PRICE_DETECTION_MODEL_PATH = os.path.join(SAVED_MODELS, "price_detection_model.pt")


# TRAINING RESULTS
TRAINING_RESULTS_FILE = os.path.join(DATA_FOLDER, "training_results.csv")
