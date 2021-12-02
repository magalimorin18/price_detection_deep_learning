# +
"""Finetuning of an RCNN for price detection."""
# %load_ext autoreload
# %autoreload 2

import os
import sys

import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.data.price_locations import PriceLocationsDataset
from src.display.display_image import display_image

sys.path.append(os.path.abspath(".."))

# -

dataset = PriceLocationsDataset()

fig, ax = plt.subplots(1, 1, figsize=(30, 15))
ele_img, ele_annotations = dataset.get_all_for_one_image("0001.jpg")
ele_annotations["price"] = "1"
display_image(ele_img, ele_annotations)

# ## Loading the model and setup

NUM_CLASSES = 2  # Price label + background

# +


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# -

# Change the head of the model with a new one, adapted to our number of classes
model.roi_heads.box_predictor = FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES
)
print(
    "Number of input features for the classes classifier",
    model.roi_heads.box_predictor.cls_score.in_features,
)

model.eval()
model(ele_img.unsqueeze(0))
