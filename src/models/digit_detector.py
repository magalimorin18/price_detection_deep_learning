"""Digit Detector."""

import logging

import cv2
import numpy as np
import pandas as pd


class DigitDetector:
    """Digit Detector."""

    def __init__(self, area_threshold=25):
        """Init."""
        logging.info("[DigitDetector] Now ready to operate!")
        self.area_threshold = area_threshold

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
                x, y, w, h = cv2.boundingRect(contour)
                results.append(
                    {
                        "x1": x,
                        "y1": y,
                        "x2": x + w,
                        "y2": y + h,
                    }
                )
        return pd.DataFrame.from_records(results)

    @staticmethod
    def prepare_img_for_detection(
        image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """Prepare image for digit detection."""
        logging.info("[DigitDetector] Preparing image for digit detection...")
        img = image[y1:y2, x1:x2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
        img = cv2.erode(img, np.ones((3, 3), np.uint8), iterations=1)
        return img
