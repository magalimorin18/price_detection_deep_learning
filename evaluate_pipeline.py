"""Evaluate pipeline."""

import logging
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.models.price_detector import PriceDetector
from src.models.price_predictor import PricePredictor
from src.processing.overlap import remove_overlaping_tags
from src.utils.distances import (
    compute_price_positions,
    compute_product_positions,
    find_closest_price,
)

# Constants
DATASET = "train"
SAVE_FIG_LOCATIONS = "./tmp/tags"
# SAVE_FIG_LOCATIONS = None

logging.basicConfig(level=logging.INFO)

# Objects
price_detector = PriceDetector()
price_predictor = PricePredictor()
annotations = pd.read_csv(f"./data/{DATASET}/annotations.csv")


def get_one_image_tags(image_path: str):
    """Get one image tags."""
    return annotations[annotations.img_name == os.path.basename(image_path)].copy()


def predict_one_image(image_path: str):
    """Predict one image results."""
    # Get the expected results
    products = get_one_image_tags(image_path)

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Get the price tags locations
    prices = price_detector.extract_prices_locations([image])[0]
    prices = remove_overlaping_tags(products, prices)

    # Find the closest price for each product
    prices = compute_price_positions(prices)
    products = compute_product_positions(products)
    product_to_price_list = []
    for _, product in products.iterrows():
        product_to_price_list.append(find_closest_price(product, prices))
    product_to_price = pd.DataFrame.from_records(product_to_price_list)

    # Find th prices values
    tag_images = []
    np_image = np.array(image)
    for i, tag in product_to_price.iterrows():
        tag_images.append(
            np_image[int(tag.price_y1) : int(tag.price_y2), int(tag.price_x1) : int(tag.price_x2)]
        )
    try:
        predicted_prices = price_predictor.extract_prices_locations(tag_images)
        if SAVE_FIG_LOCATIONS is not None:
            for i, (tag_image, pred_price) in enumerate(zip(tag_images, predicted_prices)):
                fig = plt.figure()
                plt.imshow(tag_image)
                plt.savefig(
                    os.path.join(
                        SAVE_FIG_LOCATIONS,
                        os.path.basename(image_path).split(".")[0] + f"_tag{i}_{pred_price}.png",
                    )
                )
                plt.close(fig)
    except Exception as e:
        logging.warning("Error predicting price for image %s", image_path, exc_info=True)
        predicted_prices = [-1 for _ in range(len(product_to_price))]

    assert len(predicted_prices) == product_to_price.shape[0]

    assert len(products) == len(predicted_prices)
    products["pred_price"] = predicted_prices
    products["price_tag_x1"] = product_to_price["price_x1"]
    products["price_tag_x2"] = product_to_price["price_x2"]
    products["price_tag_y1"] = product_to_price["price_y1"]
    products["price_tag_y2"] = product_to_price["price_y2"]

    for i, row in products.iterrows():
        print(
            f"Truth price: {str(row.price).ljust(5)}, Predicted price: {str(row.pred_price).ljust(5)}"
        )
    return products


if __name__ == "__main__":
    results = []
    for i, image_path in enumerate(
        tqdm(list(glob(os.path.join("data", DATASET, "images", "*.jpg")))[100:200])
    ):
        result = predict_one_image(image_path)
        results.append(result)

    results_df = pd.concat(results)
    print(f"Results shape: {results_df.shape}")
    results_df.to_csv("./results_pipeline.csv")
