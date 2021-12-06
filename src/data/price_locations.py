"""Price location."""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib.image import imread
from PIL import Image

from src.config import TRAIN_IMAGES, TRAIN_PRICE_LOCATIONS
from src.processing.vott import load_vott_data


class PriceLocationsDataset:
    """Price location dataset (where the price are located on the big image)."""

    annotations: pd.DataFrame

    def __init__(self, transforms=None) -> None:
        """Init."""
        self.annotations = load_vott_data(TRAIN_PRICE_LOCATIONS)
        self.annotations.img_name = self.annotations.img_name.apply(
            lambda name: os.path.join(TRAIN_IMAGES, name)
        )
        self.unique_images = self.annotations.img_name.unique().tolist()
        self.transforms = transforms
        logging.info(
            "The price location dataset is now loaded (%s elements)", len(self.unique_images)
        )

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.unique_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get one item."""
        img_name = self.unique_images[idx]

        # Get the image
        img = Image.open(img_name).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        # Get the target values
        rows = self.annotations[self.annotations.img_name == img_name]
        coordinates = torch.Tensor(
            [rows.x1.tolist(), rows.y1.tolist(), rows.x2.tolist(), rows.y2.tolist()]
        ).transpose(0, 1)
        target = {
            "boxes": coordinates,
            "labels": torch.ones(coordinates.shape[0], dtype=torch.int64),
        }

        return img, target

    def get_all_annotations_for_one_image(self, image_name: str) -> pd.DataFrame:
        """Get all annotations for one image."""
        annotations = self.annotations[
            self.annotations["img_name"].apply(lambda x: x.endswith(image_name))
        ]
        return annotations

    @staticmethod
    def get_original_image(image_name: str) -> np.ndarray:
        """Get one image."""
        return imread(os.path.join(TRAIN_IMAGES, image_name))

    def get_image(self, image_name: str) -> torch.Tensor:
        """Get one image."""
        img = Image.open(os.path.join(TRAIN_IMAGES, image_name)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img


if __name__ == "__main__":
    dataset = PriceLocationsDataset()
    print(dataset.annotations.head())
    print(dataset[0])
