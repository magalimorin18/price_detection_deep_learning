"""Price detector model."""

import logging
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.config import PRICE_DETECTION_MODEL_PATH
from src.utils.price_detection_utils import convert_model_output_to_format

NUM_CLASSES = 2

transforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# pylint: disable=too-few-public-methods
class PriceDetector:
    """Price detector using a fasterRCNN model."""

    def __init__(self, device: int = -1):
        """Init."""
        logging.info("[Price Detector] Initializing...")
        self.device = device
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            self.model.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES
        )
        self.model.load_state_dict(torch.load(PRICE_DETECTION_MODEL_PATH))
        self.model.to(self.device)
        logging.info("[Price Detector] Initialized.")

    def extract_prices_locations(self, images: List[np.ndarray]) -> List[pd.DataFrame]:
        """Extract prices locations from images."""
        logging.info("[Price Detector] Extracting prices locations...")
        self.model.eval()
        results = self.model(torch.tensor(np.stack(images)).to(self.device))
        prices_locations = []
        for result in results:
            model_annotations = convert_model_output_to_format(result)
            model_annotations["price"] = model_annotations["score"].apply(lambda x: round(x, 2))
            prices_locations.append(model_annotations)
        logging.info("[Price Detector] Extracted prices locations.")
        return prices_locations
