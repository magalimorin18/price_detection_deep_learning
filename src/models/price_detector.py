"""Price detector model."""

import logging
from typing import List, Union

import numpy as np
import pandas as pd
import torch

from src.config import PRICE_DETECTION_MODEL_PATH
from src.models.utils import get_model, transforms
from src.utils.price_detection_utils import convert_model_output_to_format


# pylint: disable=too-few-public-methods
class PriceDetector:
    """Price detector using a fasterRCNN model."""

    def __init__(self, device: Union[int, torch.device] = 0):
        """Init."""
        logging.info("[Price Detector] Initializing...")
        self.device = device
        self.model = get_model(model_type="resnet50", pretrained=False)
        self.model.load_state_dict(torch.load(PRICE_DETECTION_MODEL_PATH))
        # self.model.to(self.device)
        logging.info("[Price Detector] Initialized.")

    def extract_prices_locations(self, images: List[np.ndarray]) -> List[pd.DataFrame]:
        """Extract prices locations from images."""
        logging.info("[Price Detector] Extracting prices locations...")
        self.model.eval()
        prices_locations = []
        for image in images:
            result = self.model(transforms(image).unsqueeze(0))[0]
            model_annotations = convert_model_output_to_format(result)
            model_annotations["price"] = model_annotations["score"].apply(lambda x: round(x, 2))
            prices_locations.append(model_annotations)
        logging.info("[Price Detector] Extracted prices locations.")
        return prices_locations
