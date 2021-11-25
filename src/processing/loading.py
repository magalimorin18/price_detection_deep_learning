"""Loading the data."""

import os
from glob import glob
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import image

from src.config import Config


def get_file_from_id(idx: Union[int, str]) -> str:
    """Get the file name from the id."""
    return f"{str(idx).rjust(4, '0')}.jpg"


class AnnotationHandler:
    """Handle the annotations."""

    df: pd.DataFrame

    def __init__(
        self, path: str = os.path.join(Config.DATA_FOLDER, "annotations.csv")
    ) -> None:
        """Init."""
        self.df = pd.read_csv(path, index_col=0)

    def __getitem__(self, idx: Union[int, str] == 0) -> pd.DataFrame:
        """Get the annotations for one image."""
        return self.df[self.df["img_name"] == get_file_from_id(idx)].drop(
            columns=["img_name"]
        )


class ImageDataset:
    """Image dataset."""

    folder: str
    len: int

    def __init__(
        self, folder: str = os.path.join(Config.DATA_FOLDER, "images")
    ) -> None:
        """Init."""
        self.folder = folder
        self.len = len(glob(os.path.join(self.folder, "*.jpg")))

    def __len__(self) -> int:
        """Return number of images."""
        return self.len

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get one image."""
        path = os.path.join(self.folder, get_file_from_id(idx))
        if os.path.isfile(path):
            return image.imread(path)
        else:
            raise ValueError("Image not found with this idx: '%s'" % path)
