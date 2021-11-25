"""Configuration file."""

import os


class Config:
    """Configuration file."""

    SRC_FOLDER = os.path.dirname(os.path.abspath(__file__))
    ROOT_FOLDER = os.path.join(SRC_FOLDER, "..")
    DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
