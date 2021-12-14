"""Streamlit app for demonstrations."""

from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from src.display.display_image import display_annotations, display_image
from src.models.object_detector import ObjectDetector
from src.models.price_detector import PriceDetector


@st.cache(max_entries=1)
def load_objects():
    """Load the different models."""
    return ObjectDetector(), PriceDetector()


object_detector, price_detector = load_objects()


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
        products = object_detector.extract_objects([temp_image.name])

        st.markdown("Here are the detected products:")
        st.dataframe(products)

        fig, ax = plt.subplots(figsize=(10, 10))
        display_image(image, ax=ax)
        display_annotations(products, ax=ax)
        st.pyplot(fig=fig)

        # Detect the prices on the image
        st.markdown("## 3. Detect the prices")
        prices = price_detector.extract_prices_locations([image])[0]
        st.dataframe(prices)

        fig, ax = plt.subplots(figsize=(10, 10))
        display_image(image, ax=ax)
        display_annotations(prices, ax=ax)
        st.pyplot(fig=fig)
