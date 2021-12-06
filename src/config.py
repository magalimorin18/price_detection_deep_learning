"""Configuration file."""

import os


# pylint: disable=too-few-public-methods
class Config:
    """Configuration file."""

    SRC_FOLDER = os.path.dirname(os.path.abspath(__file__))
    ROOT_FOLDER = os.path.join(SRC_FOLDER, "..")
    TRAIN_FOLDER = os.path.join(ROOT_FOLDER, "data", "train")
    TEST_FOLDER = os.path.join(ROOT_FOLDER, "data", "test")
