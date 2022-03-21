"""Streamlit app for demonstrations."""

from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.display.display_image import display_annotations, display_image
from src.models.object_detector import ObjectDetector
from src.models.price_detector import PriceDetector
from src.models.price_predictor import PricePredictor
from src.utils.distances import (
    compute_price_positions,
    compute_product_positions,
    find_closest_price,
)


@st.cache(max_entries=1)
def load_objects():
    """Load the different models."""
    return ObjectDetector(), PriceDetector(), PricePredictor()


object_detector, price_detector, price_predictor = load_objects()


st.markdown("# Price detection")

st.markdown(
    "Price detection in supermarkets to quickly identify the prices of the different products."
)


st.markdown("## 1. Upload an image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

with NamedTemporaryFile(delete=False) as temp_image:

    if uploaded_file is not None:
        st.markdown("Your image has been uploaded.")

        temp_image.write(uploaded_file.getvalue())
        image = Image.open(temp_image.name).convert("RGB")
        st.image(uploaded_file, width=200)

        # Detect the products on the image
        st.markdown("## 2. Detect the products")
        with st.spinner("Finding products..."):
            products = object_detector.extract_objects([temp_image.name])

        st.markdown("Here are the detected products:")
        st.dataframe(products)

        fig, ax = plt.subplots(figsize=(10, 10))
        display_image(image, ax=ax)
        display_annotations(products, ax=ax)
        st.pyplot(fig=fig)

        # Detect the prices on the image
        st.markdown("## 3. Detect the prices")
        with st.spinner("Finding prices..."):
            prices = price_detector.extract_prices_locations([image])[0]
        st.dataframe(prices)

        # TODO: Remove the prices that overlap more than 50% with the products

        fig, ax = plt.subplots(figsize=(10, 10))
        display_image(image, ax=ax)
        display_annotations(prices, ax=ax, color=0)
        display_annotations(products, ax=ax, color=1)
        st.pyplot(fig=fig)

        st.markdown("## 4. Find the corresponding price for each product")
        with st.spinner("Finding the right prices..."):
            prices = compute_price_positions(prices)
            products = compute_product_positions(products)
            product_to_price_list = []
            for i, product in products.iterrows():
                product_to_price_list.append(find_closest_price(product, prices))
            product_to_price = pd.DataFrame.from_records(product_to_price_list)
        st.dataframe(product_to_price)

        fig, ax = plt.subplots(figsize=(10, 10))
        display_image(image, ax=ax)
        display_annotations(prices, ax=ax, color=0)
        display_annotations(products, ax=ax, color=1)
        for product in product_to_price.itertuples():
            ax.plot([product.pos_x, product.price_x], [product.pos_y, product.price_y], c="b")
        st.pyplot(fig=fig)

        st.markdown("## 5. Predict the price of each product")
        with st.spinner("Predicting prices..."):
            tag_images = []
            np_image = np.array(image)
            for _, tag in prices.iterrows():
                tag_images.append(np_image[int(tag.y1) : int(tag.y2), int(tag.x1) : int(tag.x2)])
            predicted_prices = price_predictor.extract_prices_locations(tag_images)
            prices["price"] = predicted_prices

        fig, ax = plt.subplots(figsize=(10, 10))
        display_image(image, ax=ax)
        display_annotations(prices, ax=ax, color=0)
        display_annotations(products, ax=ax, color=1)
        for product in product_to_price.itertuples():
            ax.plot([product.pos_x, product.price_x], [product.pos_y, product.price_y], c="b")
        st.pyplot(fig=fig)
