"""
Digit recognition.
Création d'une classe DigitRecognitionDataset : Crée un dataset à partir des csv de digits

"""

import logging
import os
from typing import Literal, Tuple

import cv2
import pandas as pd
import torch
from matplotlib.image import imread

from src.config import (
    TEST_DIGIT_LOCATIONS_CSV_FOLDER,
    TEST_DIGIT_LOCATIONS_IMAGES_FOLDER,
    TRAIN_DIGIT_LOCATIONS_CSV_FOLDER,
    TRAIN_DIGIT_LOCATIONS_IMAGES_FOLDER,
    VAL_DIGIT_LOCATIONS_CSV_FOLDER,
    VAL_DIGIT_LOCATIONS_IMAGES_FOLDER,
)


class DigitRecognitionDataset:
    """Price Recognition dataset"""

    annotations: pd.DataFrame  # c'est du typage

    def __init__(self, dataset: Literal["train", "test", "val"] = "train", transforms=None) -> None:
        """Init."""
        if dataset == "train":
            self.image_folder = TRAIN_DIGIT_LOCATIONS_IMAGES_FOLDER
            self.csv_folder = TRAIN_DIGIT_LOCATIONS_CSV_FOLDER

        elif dataset == "test":
            self.image_folder = TEST_DIGIT_LOCATIONS_IMAGES_FOLDER
            self.csv_folder = TEST_DIGIT_LOCATIONS_CSV_FOLDER

        elif dataset == "val":
            self.image_folder = VAL_DIGIT_LOCATIONS_IMAGES_FOLDER
            self.csv_folder = VAL_DIGIT_LOCATIONS_CSV_FOLDER

        list_lien = os.listdir(
            self.csv_folder
        )  # liste de tous les liens qui sont dans la dossier image
        self.annotations = pd.concat(
            [pd.read_csv(os.path.join(self.csv_folder, lien)) for lien in list_lien]
        )
        self.transforms = transforms
        logging.info(
            "The digit recognition dataset is now loaded (%s elements)", len(self.annotations)
        )

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Entry: indentifiant unique d'un digit.
        Returns (image, label)
        image : image taille du MNIST, en noir et blanc, croppé sur le digit, format torch tensor
        label : numéro du digit
        """
        rows = self.annotations.iloc[idx]
        img_tag_name = rows.img_tag
        # Get the image
        img_tag = imread(os.path.join(self.image_folder, str(img_tag_name)))

        # Get the target values
        coordinates = [rows.x1, rows.y1, rows.x2, rows.y2]

        img_digit = img_tag[coordinates[1] : coordinates[3], coordinates[0] : coordinates[2]]
        img_black_and_white = cv2.cvtColor(img_digit, cv2.COLOR_BGR2GRAY)
        img_invert_color = cv2.bitwise_not(img_black_and_white)
        _, thresh = cv2.threshold(img_invert_color, 100, 255, 0)
        thresh = cv2.resize(thresh, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

        if self.transforms is not None:
            thresh = self.transforms(thresh)

        target = rows.label

        return thresh, torch.tensor([target])
