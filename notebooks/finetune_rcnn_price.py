"""Finetuning of an RCNN for price detection.

Inspiration from this tutorial from pytorch:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
# pylint: disable=wrong-import-position,invalid-name,expression-not-assigned

# %load_ext autoreload
# %autoreload 2

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(".."))
from src.config import PRICE_DETECTION_MODEL_PATH
from src.data.price_locations import PriceLocationsDataset
from src.display.display_image import display_annotations, display_image
from src.models.utils import get_model, train_one_epoch, transforms
from src.utils.price_detection_utils import convert_model_output_to_format

logging.basicConfig(level=logging.INFO)
# -
print(transforms)

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
model = get_model(model_type="resnet50")
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

# To check the model output when in train mode
model.train()
img, annotations = dataset[0]
print(annotations.keys(), annotations["boxes"].shape, annotations["labels"].shape)
model(img.unsqueeze(0), [annotations])

# +
# Params before training
EPOCHS = 3
BATCH_SIZE = 1
OPTI_NAME = "SGD"
OPTI_LEARNING_RATE = 0.005
OPTI_MOMENTUM = 0.9
OPTI_WEIGHT_DECAY = 0.0005
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=OPTI_LEARNING_RATE, momentum=OPTI_MOMENTUM, weight_decay=OPTI_WEIGHT_DECAY
)
# -

train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)

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

# +
# Save the model
torch.save(model.state_dict(), PRICE_DETECTION_MODEL_PATH)

from src.models.utils import evaluate_and_save

evaluate_and_save(
    model,
    train_loader,
    device=DEVICE,
    params=dict(
        OPTI_LEARNING_RATE=OPTI_LEARNING_RATE,
        OPTI_MOMENTUM=OPTI_MOMENTUM,
        OPTI_WEIGHT_DECAY=OPTI_WEIGHT_DECAY,
        BATCH_SIZE=BATCH_SIZE,
        EPOCHS=EPOCHS,
    ),
)
# -

# ## Analysis of the results

# Load the model
model = get_model(model_type="resnet50", pretrained=False)
model.load_state_dict(torch.load(PRICE_DETECTION_MODEL_PATH))

# +
model.to("cpu")
img_path = "../../test/images/0651.jpg"


# Compute the model annotations
model.eval()
results = model(dataset.get_image(img_path).unsqueeze(0))
model_annotations = convert_model_output_to_format(results[0])
model_annotations["price"] = model_annotations["score"].apply(lambda x: round(x, 2))
print(model_annotations)
# Display
fig, ax = plt.subplots(1, 1, figsize=(30, 15))
display_image(dataset.get_original_image(img_path), ax=ax)
# display_annotations(ele_annotations, ax=ax, color=0)
display_annotations(model_annotations, ax=ax, color=1)
# -


{k: np.mean(v) for k, v in results.items()}
