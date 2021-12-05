# +
"""Finetuning of an RCNN for price detection.

Inspiration from this tutorial from pytorch:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
# pylint: disable=wrong-import-position
# %load_ext autoreload
# %autoreload 2

import logging
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.append(os.path.abspath(".."))
from src.config import SAVED_MODELS
from src.data.price_locations import PriceLocationsDataset
from src.display.display_image import display_annotations, display_image
from src.models.object_detector import train_one_epoch
from src.utils.price_detection_utils import convert_model_output_to_format

logging.basicConfig(level=logging.INFO)
# -

transforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = PriceLocationsDataset(transforms=transforms)

ele_img, ele_annotations = dataset[0]
print(ele_img.shape)
x = ele_img.flatten()
print(x.max(), x.min())

# +
img_path = os.path.basename(dataset.annotations.img_name.sample(1).iloc[0])

# Get true annotations
ele_img = dataset.get_original_image(img_path)
ele_annotations = dataset.get_all_annotations_for_one_image(img_path)

# Display the image and annotations
fig, ax = plt.subplots(1, 1, figsize=(30, 15))
display_image(ele_img, ax=ax)
display_annotations(ele_annotations, ax=ax, color=0)
# -

# ## Loading the model and setup

NUM_CLASSES = 2  # Price label + background

# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Change the head of the model with a new one, adapted to our number of classes
model.roi_heads.box_predictor = FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES
)
print(
    "Number of input features for the classes classifier",
    model.roi_heads.box_predictor.cls_score.in_features,
)

# +
img_path = os.path.basename(dataset.annotations.img_name.sample(1).iloc[0])

# Get true annotations
ele_annotations = dataset.get_all_annotations_for_one_image(img_path)

# Compute the model annotations
model.eval()
results = model(dataset.get_image(img_path).unsqueeze(0))
model_annotations = convert_model_output_to_format(results[0])

# Display
fig, ax = plt.subplots(1, 1, figsize=(30, 15))
display_image(dataset.get_original_image(img_path), ax=ax)
display_annotations(ele_annotations, ax=ax, color=0)
display_annotations(model_annotations, ax=ax, color=1)
# -

model.train()
img, annotations = dataset[0]
print(annotations.keys(), annotations["boxes"].shape, annotations["labels"].shape)
model(img.unsqueeze(0), [annotations])

# +
# Params before training
EPOCHS = 3
BATCH_SIZE = 1
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# -

# +


train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
# -

model.to(DEVICE)
for epoch in range(EPOCHS):
    train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)

# +
model.to("cpu")
img_path = os.path.basename(dataset.annotations.img_name.sample(1).iloc[0])

# Get true annotations
ele_annotations = dataset.get_all_annotations_for_one_image(img_path)

# Compute the model annotations
model.eval()
results = model(dataset.get_image(img_path).unsqueeze(0))
model_annotations = convert_model_output_to_format(results[0])
print(model_annotations)
# Display
fig, ax = plt.subplots(1, 1, figsize=(30, 15))
display_image(dataset.get_original_image(img_path), ax=ax)
display_annotations(ele_annotations, ax=ax, color=0)
display_annotations(model_annotations, ax=ax, color=1)
# -

# Save the model
torch.save(model.state_dict(), os.path.join(SAVED_MODELS, "price_detection"))
