"""Price location."""

import logging
import os
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib.image import imread
from PIL import Image

from src.config import TEST_IMAGES, TRAIN_IMAGES
from src.processing.annotations import load_price_annotations


class PriceLocationsDataset:
    """Price location dataset (where the price are located on the big image)."""

    annotations: pd.DataFrame

    def __init__(self, transforms=None, dataset: Literal["train", "test"] = "train") -> None:
        """Init."""
        self.image_folder = TRAIN_IMAGES if dataset == "train" else TEST_IMAGES
        self.annotations = load_price_annotations()
        self.annotations.img_name = self.annotations.img_name.apply(
            lambda name: os.path.join(self.image_folder, name)
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

    def get_original_image(self, image_name: str) -> np.ndarray:
        """Get one image."""
        return imread(os.path.join(self.image_folder, image_name))

    def get_image(self, image_name: str) -> torch.Tensor:
        """Get one image."""
        img = Image.open(os.path.join(self.image_folder, image_name)).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img


if __name__ == "__main__":
    dataset_ = PriceLocationsDataset()
    print(dataset_.annotations.head())
    print(dataset_[0])
