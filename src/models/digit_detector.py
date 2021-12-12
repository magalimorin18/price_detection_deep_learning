"""Digit Detector."""

import logging
import os
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms

from src.config import SAVED_MODELS

MODEL_PATH = os.path.join(SAVED_MODELS, "digit_detector.pt")
digit_classifier = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(1568, 10),
)

digits_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


class DigitDetector:
    """Digit Detector."""

    def __init__(self, area_threshold=25):
        """Init."""
        self.area_threshold = area_threshold
        self.model = digit_classifier
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        logging.info("[DigitDetector] Now ready to operate!")

    # pylint: disable=too-many-locals
    def extract_digits_locations(self, image: np.ndarray) -> pd.DataFrame:
        """Extract digits locations from one image."""
        logging.info("[DigitDetector] Extracting digits from image...")
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_image = cv2.GaussianBlur(grey_image, (5, 5), 0)
        thresh_image = cv2.adaptiveThreshold(blur_image, 255, 1, 1, 11, 3)
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.area_threshold:
                x1, y1, w, h = cv2.boundingRect(contour)
                x2, y2 = x1 + w, y1 + h
                results.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "result": self.classify_image(image, x1, y1, x2, y2),
                    }
                )
        return pd.DataFrame.from_records(results)

    @staticmethod
    def prepare_img_for_detection(
        image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """Prepare image for digit detection."""
        img = image[y1:y2, x1:x2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = cv2.bitwise_not(img)
        img = (img > 130).astype(float)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = digits_transform(img)
        return img

    # pylint: disable=too-many-arguments
    def classify_image(
        self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> Union[int, float]:
        """Classify one image."""
        image = self.prepare_img_for_detection(image, x1, y1, x2, y2)
        with torch.no_grad():
            output = self.model(image.unsqueeze(0).float())
        return output.argmax().item(), round(output.max().item(), 2)
