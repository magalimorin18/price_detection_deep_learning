"""Price location."""

import os
from typing import Tuple

import pandas as pd
import torch
from matplotlib.image import imread

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
        self.transforms = transforms

    def __len__(self) -> int:
        """Length of the dataset."""
        return self.annotations.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get one item."""
        row = self.annotations.iloc[idx]
        img = torch.IntTensor(imread(row.img_name))
        coordinates = torch.Tensor([row.x1, row.y1, row.x2, row.y2])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, coordinates

    def get_all_for_one_image(self, image_name: str) -> Tuple[torch.Tensor, pd.DataFrame]:
        """Get all annotations for one image."""
        annotations = self.annotations[
            self.annotations["img_name"].apply(lambda x: x.endswith(image_name))
        ]
        img = torch.IntTensor(imread(os.path.join(TRAIN_IMAGES, image_name)))
        return img, annotations


if __name__ == "__main__":
    dataset = PriceLocationsDataset()
    print(dataset.annotations.head())
    print(dataset[0])
