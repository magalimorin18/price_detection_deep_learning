"""Price prediction"""
# pylint:disable=R0903
# Too few public methods (1/2) (too-few-public-methods)
import logging
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import patches
from skimage.transform import resize
from torchvision import transforms
from tqdm import tqdm

from src.config import PRICE_PREDICTOR_MODEL_PATH
from src.data.price_locations import PriceLocationsDataset
from src.models.class_cnnnet import CNNNet


class PricePredictor:
    """Price prediction using CNN model"""

    CONTOUR_BORDER = 2
    GRAY_THRESHOLD = 100
    SIZE_RATIO = 0.6
    MIN_WIDTH = 11
    MIN_HEIGHT = 11

    def __init__(self, device: Union[int, torch.device] = 0):
        """Init."""
        logging.info("[Price Predictor] Initializing...")
        self.device = device
        self.model = CNNNet()
        self.model.load_state_dict(torch.load(PRICE_PREDICTOR_MODEL_PATH))
        # self.model.to(self.device)
        logging.info("[Price Predictor] Initialized.")

    def extract_prices_locations(self, tag_images: List[np.ndarray]) -> List[float]:
        """Extract prices from a tag"""
        logging.info("[Price Predictor] Extracting prices prediction...")
        self.model.eval()
        prices_predictions: List[float] = []
        for tag_image in tqdm(tag_images, desc="Extract and classify digits"):
            result = self._extract_prices_locations_one(tag_image)
            prices_predictions.append(self._convert_output_to_number(result))
        logging.debug(prices_predictions)
        logging.info("[Price Detector] Extracted prices locations.")
        return prices_predictions

    def _extract_prices_locations_one(self, tag_image: np.ndarray) -> str:
        """
        Entry: The tag image
        Returns : The price on the tag
        """
        # Detect all the characters on the image
        list_digits_img = []

        # Transformations (black and white)
        img_black_and_white = cv2.cvtColor(tag_image, cv2.COLOR_BGR2GRAY)
        img_invert_color = cv2.bitwise_not(img_black_and_white)
        _, thresh = cv2.threshold(img_invert_color, self.GRAY_THRESHOLD, 255, 0)

        # Detect the digits contours
        list_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            df_digits_contours = self._convert_cv2_contours_to_dataframe(list_contours)
        except AttributeError:
            logging.error(
                "Seems like we did not found any digit on the tag... (nbr tags: %i)",
                len(list_contours),
            )
            return ""
        # df_digits_contours = self._filter_contours(df_digits_contours, distance_btw_digits=2)
        # self._plot_with_contours(thresh, df_digits_contours)

        # Crop the digits
        for i in range(len(df_digits_contours)):
            df_single_digit = df_digits_contours.iloc[i]
            digit_img = thresh[
                int(df_single_digit.y1) : int(df_single_digit.y2),
                int(df_single_digit.x1) : int(df_single_digit.x2),
            ]
            list_digits_img.append(digit_img)
        logging.debug("[Price Predictor] Detected %d digits", len(list_digits_img))

        # Classify all those characters
        outputs: List[int] = []
        for digit_image in list_digits_img:
            outputs.append(self._classify_digit(digit_image)[0])

        return "".join(list(map(str, outputs)))

    def _classify_digit(self, digit_img: np.ndarray, threshold_proba=0.1) -> Tuple[int, float]:
        """
        Entry : the image of a digit
        Returns : the predicted digit as an int
        """
        digit_tensor = self._digit_img_transformation(digit_img)
        digit_prediction_tensor = self.model(digit_tensor.unsqueeze(0))[0]
        digit_prediction_tensor = torch.nn.Softmax(dim=0)(digit_prediction_tensor)
        digit_prediction = digit_prediction_tensor.argmax()
        digit_prediction_int = int(digit_prediction.numpy())
        proba = digit_prediction_tensor[digit_prediction_int].item()
        if proba > threshold_proba:
            return digit_prediction_int, proba
        return -1, 0.0

    @staticmethod
    def _digit_img_transformation(digit_img: np.ndarray, padding=10):
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

    def _convert_cv2_contours_to_dataframe(self, list_contours: List) -> pd.DataFrame:
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
                    "x1": box[0][0] - self.CONTOUR_BORDER,
                    "y1": box[0][1] - self.CONTOUR_BORDER,
                    "x2": box[2][0] + self.CONTOUR_BORDER,
                    "y2": box[2][1] + self.CONTOUR_BORDER,
                }
            )

        # Convert to dataframe and remove useless rectangles
        tag_contours = pd.DataFrame.from_records(rect_list)
        tag_contours = tag_contours[
            (tag_contours > 0).all(axis=1)
            & (tag_contours.x2 - tag_contours.x1 > self.MIN_WIDTH)
            & (tag_contours.y2 - tag_contours.y1 > self.MIN_HEIGHT)
            & (
                (tag_contours.x2 - tag_contours.x1) * self.SIZE_RATIO
                < (tag_contours.y2 - tag_contours.y1)
            )
        ]

        return tag_contours.copy().sort_values(by=["x1", "y1"])

    @staticmethod
    def _filter_contours(df: pd.DataFrame, distance_btw_digits: int = 2):
        """
        Entry : dataframe with delimitation of each contour (x1,x2,y1,y2)
        Returns : filter the contours that cannot logically be digits
        """
        boxes = []  # Liste des indices du dataframe ou il y a les bon digits
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

    @staticmethod
    def _convert_output_to_number(output: str) -> float:
        """Convert the output to a number."""
        if len(output) == 0:
            return -1.0
        if len(output) <= 2:
            return float(output)
        if len(output) == 3:
            return float(f"{output[:2]}.{output[2]}")
        return float(f"{output[:2]}.{output[2:4]}")

    @staticmethod
    def _plot_with_contours(img: np.ndarray, contours: pd.DataFrame = None):
        _, ax = plt.subplots()
        ax.imshow(img)
        if contours is not None:
            for _, tag_contour in contours.iterrows():
                ax.add_patch(
                    patches.Rectangle(
                        (tag_contour["x1"], tag_contour["y1"]),
                        tag_contour["x2"] - tag_contour["x1"],
                        tag_contour["y2"] - tag_contour["y1"],
                        color="b",
                        fill=None,
                    )
                )
        plt.show()


if __name__ == "__main__":
    dataset = PriceLocationsDataset()
    df_tag = pd.read_csv("./data/train/price_tags.csv").iloc[1]
    image = dataset.get_original_image(df_tag.img_name)
    tag_img = image[int(df_tag.y1) : int(df_tag.y2), int(df_tag.x1) : int(df_tag.x2)]
    price_predictor = PricePredictor()
    predicted_price = price_predictor.extract_prices_locations([tag_img])
    print(predicted_price)
