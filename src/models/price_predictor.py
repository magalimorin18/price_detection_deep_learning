"""Price prediction"""
# pylint:disable=R0903
# Too few public methods (1/2) (too-few-public-methods)
import logging
from typing import List, Union

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.transform import resize
from torchvision import transforms
from tqdm import tqdm

from src.config import PRICE_PREDICTOR_MODEL_PATH
from src.models.class_cnnnet import CNNNet


class PricePredictor:
    """Price prediction using CNN model"""

    def __init__(self, device: Union[int, torch.device] = 0):
        """Init."""
        logging.info("[Price Predictor] Initializing...")
        self.device = device
        self.model = CNNNet()
        self.model.load_state_dict(torch.load(PRICE_PREDICTOR_MODEL_PATH))
        # self.model.to(self.device)
        logging.info("[Price Predictor] Initialized.")

    def extract_prices_locations(self, tag_images: List[np.ndarray]) -> List[str]:
        """Extract prices from a tag"""
        logging.info("[Price Predictor] Extracting prices prediction...")
        self.model.eval()
        prices_prediction: List[str] = []
        for tag_image in tqdm(tag_images, desc="Extract and classify digits"):
            result = self.__extract_prices_locations_one(tag_image)
            prices_prediction.append(result)
        logging.info("[Price Detector] Extracted prices locations.")
        return prices_prediction

    def __extract_prices_locations_one(self, tag_image: np.ndarray) -> str:
        """
        Entry: The tag image
        Returns : The price on the tag
        """
        # Detect all the characters on the image
        digit_images = self.__detect_digits_on_image(tag_image)

        # Classify all those characters
        outputs: List[int] = []
        for digit_image in digit_images:
            outputs.append(self.__classify_digit(digit_image))

        return "".join(list(map(str, outputs)))

    def __classify_digit(self, digit_img: np.ndarray, threshold_proba=0.1) -> int:
        """
        Entry : the image of a digit
        Returns : the predicted digit as an int
        """
        digit_tensor = self.__digit_img_transformation(digit_img)
        digit_prediction_tensor = self.model(digit_tensor.unsqueeze(0))[0]
        digit_prediction_tensor = torch.nn.Softmax()(digit_prediction_tensor)
        digit_prediction = digit_prediction_tensor.argmax()
        digit_prediction_int = int(digit_prediction.numpy())
        proba = digit_prediction_tensor[digit_prediction_int].item()
        if proba > threshold_proba:
            return digit_prediction_int
        return -1

    @staticmethod
    def __digit_img_transformation(digit_img: np.ndarray, padding=10):
        """
        Entry : image of a digit
        Returns : image of the digit transformed to the format of a tensor
        """
        digit_reshaped = np.zeros(
            (digit_img.shape[0] + 2 * padding, digit_img.shape[1] + 2 * padding)
        )
        digit_reshaped[padding:-padding, padding:-padding] = digit_img
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        digit_resize = resize(digit_reshaped, (28, 28))
        digit_tensor = trans(digit_resize[:, :])
        digit_tensor = digit_tensor.float()
        digit_tensor = digit_tensor.mean(axis=0)
        digit_tensor = digit_tensor.unsqueeze(0)
        return digit_tensor

    def __detect_digits_on_image(self, tag_img: np.ndarray) -> List[np.ndarray]:
        """
        Entry : image of a tag
        Returns : a list of the images of each digits detected on the tag
        """
        list_digits_img = []

        # Transformations (black and white)
        img_black_and_white = cv2.cvtColor(tag_img, cv2.COLOR_BGR2GRAY)
        img_invert_color = cv2.bitwise_not(img_black_and_white)
        _, thresh = cv2.threshold(img_invert_color, 100, 255, 0)

        # Detect the digits contours
        list_contours, _ = cv2.findContours(tag_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        df_digits_contours = self.__create_contours(list_contours, precision=2)
        df_digits_contours = self.__filter_contours(df_digits_contours, distance_btw_digits=2)

        # Crop the digits
        for i in range(len(df_digits_contours)):
            df_single_digit = df_digits_contours.iloc[i]
            digit_img = thresh[
                int(df_single_digit.y1) : int(df_single_digit.y2),
                int(df_single_digit.x1) : int(df_single_digit.x2),
            ]
            list_digits_img.append(digit_img)
        return list_digits_img

    @staticmethod
    def __create_contours(list_contours: List, precision: int) -> pd.DataFrame:
        """
        Entry : list of the contours with width height of the digits on a tag
        Returns : dataframe with delimitation of each contour (x1,x2,y1,y2)
        """
        rect_list = []
        for contour in list_contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            rect_list.append(
                {
                    "x1": box[0][0] - precision,
                    "y1": box[0][1] - precision,
                    "x2": box[2][0] + precision,
                    "y2": box[2][1] + precision,
                }
            )
        tag_contours = pd.DataFrame.from_records(rect_list)  # dataframe avec x1, x2, y1, y2
        tag_contours_without_neg_values = tag_contours[
            (tag_contours > 0).all(axis=1)
            & (tag_contours.x2 - tag_contours.x1 > 1)
            & (tag_contours.y2 - tag_contours.y1 > 1)
        ]
        df_digits_contours = tag_contours_without_neg_values[
            ((tag_contours.x2 - tag_contours.x1) < (tag_contours.y2 - tag_contours.y1))
        ]
        return df_digits_contours

    @staticmethod
    def __filter_contours(df_digits_contours: pd.DataFrame, distance_btw_digits: int = 2):
        """
        Entry : dataframe with delimitation of each contour (x1,x2,y1,y2)
        Returns : filter the contours that cannot logically be digits
        """
        boxes = []  # Liste des indices du dataframe ou il y a les bon digits
        df = df_digits_contours.sort_values(by=["x1"], ascending=True)
        for i in range(len(df) - 1):
            for j in range(len(df) - 1):
                box1x2 = df.iloc[i, 2]
                box2x1 = df.iloc[j, 0]

                box1y1 = df.iloc[i, 1]
                box2y1 = df.iloc[j, 1]

                box1y2 = df.iloc[i, 3]
                box2y2 = df.iloc[j, 3]

                condition_1 = box2x1 - distance_btw_digits <= box1x2 <= box2x1 + distance_btw_digits
                condition_2 = (
                    box2y1 - distance_btw_digits <= box1y1 <= box2y1 + distance_btw_digits
                ) & (box2y2 - distance_btw_digits <= box1y2 <= box2y2 + distance_btw_digits)

                if condition_1 & condition_2:
                    if i not in boxes:
                        boxes.append(i)
                    if j not in boxes:
                        boxes.append(j)
        if len(boxes) == 0:
            return boxes
        df_digits_contours = df.iloc[boxes]
        return df_digits_contours
