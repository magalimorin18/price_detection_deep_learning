# -*- coding: utf-8 -*-
# +
# pylint: disable=wrong-import-position,invalid-name,pointless-statement
"""
Pipeline to predict if the price of a tag
"""
# %load_ext autoreload
# %autoreload 2


import logging
import os
import sys

sys.path.append(os.path.abspath(".."))
import cv2
import numpy as np
import pandas as pd
import torch
from skimage.transform import resize
from torchvision import transforms
from tqdm import tqdm

from src.data.price_locations import PriceLocationsDataset
from src.models.class_cnnnet import CNNNet

logging.basicConfig(level=logging.INFO)
# -

model = CNNNet()
model.load_state_dict(torch.load("finetune_OCR.pth"))

dataset = (
    PriceLocationsDataset()
)  # dataset est une instanse d'une classe et PriceLocationsDataset est une classe


# +
def pipeline_annotations(path_annotations_csv):
    """
    Entry: csv with all tag annotation on every image
    Returns : same csv with a price_predicted column
    """
    df_annotations = pd.read_csv(path_annotations_csv)
    df_annotations_predict = df_annotations.copy()
    df_annotations_predict = df_annotations_predict.loc[0:200, :]
    df_annotations_predict["price_predicted"] = 0
    with tqdm(
        total=len(df_annotations_predict), desc="Compute the influence", colour="green"
    ) as pbar:
        for i in df_annotations_predict.index:
            price = predict_price(
                df_annotations_predict.loc[i]
            )  # I give predict_price the coordinates of the tag i
            df_annotations_predict.loc[i, "price_predicted"] = price
            pbar.update()
    return df_annotations_predict


def predict_price(df_single_tag):
    """
    Entry : a pandas dataframe with the coordinates of a tag on an image
    Returns : the price detected on this tag
    """
    str_digits = []
    image = dataset.get_original_image(df_single_tag.img_name)
    # Récupérer l'image croppée
    tag_img = image[
        int(df_single_tag.y1) : int(df_single_tag.y2), int(df_single_tag.x1) : int(df_single_tag.x2)
    ]
    #     display tag image
    #     fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    #     display_image(tag_img, ax=ax)
    # Trouver les digits
    list_digits_img = digits_img(tag_img)
    for digit_img in list_digits_img:
        digit = predict_digit(digit_img)
        str_digits.append(digit)
    str_digits = [str(i) for i in str_digits]
    price = "".join(str_digits)
    return price


# +


def predict_digit(digit_img, threshold_proba=0.1):
    """
    Entry : the image of a digit
    Returns : the predicted digit as an int
    """
    digit_tensor = digit_img_transformation(digit_img)
    digit_prediction_tensor = model(digit_tensor.unsqueeze(0))[0]
    digit_prediction_tensor = torch.nn.Softmax()(digit_prediction_tensor)
    digit_prediction = digit_prediction_tensor.argmax()
    digit_prediction = digit_prediction.numpy()
    proba = digit_prediction_tensor[digit_prediction].item()
    if proba > threshold_proba:
        return digit_prediction
    return -1


def digit_img_transformation(digit_img, padding=10):
    """
    Entry : image of a digit
    Returns : image of the digit transformed to the format of a tensor
    """
    digit_reshaped = np.zeros((digit_img.shape[0] + 2 * padding, digit_img.shape[1] + 2 * padding))
    digit_reshaped[padding:-padding, padding:-padding] = digit_img
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    digit_resize = resize(digit_reshaped, (28, 28))
    digit_tensor = trans(digit_resize[:, :])
    digit_tensor = digit_tensor.float()
    digit_tensor = digit_tensor.mean(axis=0)
    digit_tensor = digit_tensor.unsqueeze(0)
    return digit_tensor


# +
def digits_img(tag_img):
    """
    Entry : image of a tag
    Returns : a list of the images of each digits detected on the tag
    """
    list_digits_img = []
    img_black_and_white = cv2.cvtColor(tag_img, cv2.COLOR_BGR2GRAY)
    img_invert_color = cv2.bitwise_not(img_black_and_white)
    _, thresh = cv2.threshold(img_invert_color, 100, 255, 0)
    df_digits_contours = detect_contours(thresh)
    for i in range(len(df_digits_contours)):
        df_single_digit = df_digits_contours.iloc[i]
        digit_img = thresh[
            int(df_single_digit.y1) : int(df_single_digit.y2),
            int(df_single_digit.x1) : int(df_single_digit.x2),
        ]
        list_digits_img.append(digit_img)
    return list_digits_img


def detect_contours(tag_img):
    """
    Entry : image of a tag
    Returns : a dataframe with the contours of the digits detected on the tag
    """
    list_contours, _ = cv2.findContours(tag_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    df_digits_contours = create_contours(list_contours, precision=2)
    df_digits_contours = filter_contours(df_digits_contours, distance_btw_digits=2)
    return df_digits_contours


def create_contours(list_contours, precision):
    """
    Entry : list of the contours with width height of the digits on a tag
    Returns : dataframe with delimitation of each contour (x1,x2,y1,y2)
    """
    rect_list = []
    x = precision
    for contour in list_contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        rect_list.append(
            {"x1": box[0][0] - x, "y1": box[0][1] - x, "x2": box[2][0] + x, "y2": box[2][1] + x}
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


def filter_contours(df_digits_contours, distance_btw_digits=2):
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


# -

annotations_price_predicted = pipeline_annotations("../data/train/price_tags.csv")

annotations_price_predicted.head(20)

annotations_price_predicted[annotations_price_predicted["price_predicted"] == ""]

# TEST ------
# +
# annotations_init = dataset.get_all_annotations_for_one_image('0001.jpg')
# annotations_init.head()

# +
# annotations_price = pipeline('0001.jpg')

# +
# annotations_price.head()
# -

df_annotations_train_price_tags = pd.read_csv("../data/train/price_tags.csv")
df_annotations_train_price_tags.head()
