"""Utils for models."""

import datetime
import logging
import os
from math import ceil
from random import choice
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from src.config import TRAINING_RESULTS_FILE
from src.data.price_locations import PriceLocationsDataset
from src.utils.metrics import metric_iou
from src.utils.price_detection_utils import convert_model_output_to_format

NUM_CLASSES = 2  # Price label + background
transforms = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_model(*, model_type: Literal["resnet50"] = "resnet50", pretrained=True):
    """Return the pretrained model."""
    if model_type == "resnet50":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    elif model_type == "mobilnetv3":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=pretrained
        )
    else:
        raise NotImplementedError(f"Model {model_type} not implemented")
    # Change the head of the model with a new one, adapted to our number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES
    )
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train one epoch.

    Inspired mainly by: https://github.com/pytorch/vision/blob/main/references/detection/engine.py
    """
    model.to(device)
    model.train()

    # WORK ONLY FOR ONE ELEMENT PER BATCH FOR NOW
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}", total=len(data_loader)):
        # Put the data on the device
        images = list(image.to(device) for image in images)

        targets = [{k: v.squeeze(0).to(device) for k, v in targets.items()}]

        # Pass on the model + Compute the loss
        optimizer.zero_grad()
        loss_dict = model(images, targets)

        losses: torch.Tensor = sum(loss for loss in loss_dict.values())

        # Compute gradient and optimize
        losses.backward()
        optimizer.step()
    return losses


def evaluate_loss(model, data_loader, device):
    """Evaluate one model on object detection."""
    model.to(device)
    losses = {
        k: []
        for k in [
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
            "iou_score",
        ]
    }
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluation", total=len(data_loader)):
            # Put the data on the device
            images = list(image.to(device) for image in images)

            targets = [{k: v.squeeze(0).to(device) for k, v in targets.items()}]

            # Pass on the model + Compute the loss
            # TODO: Super slow for now, but now way without
            # Changing how the repo work
            model.train()
            loss_dict = model(images, targets)
            model.eval()
            pred = model(images)

            # IOU score
            pred_locations = convert_model_output_to_format(pred[0])
            true_locations = convert_model_output_to_format(targets[0])
            iou_score = metric_iou(images[0], true_locations, pred_locations)

            for k, v in loss_dict.items():
                losses[k].append(v.item())
            losses["iou_score"].append(iou_score)
    return losses


def evaluate_and_save(model, data_loader, device, params):
    """Evaluate one model on object detection."""
    # Get the data
    loss_results = evaluate_loss(model, data_loader, device)
    loss_means = {k: np.mean(v) for k, v in loss_results.items()}
    result = {**loss_means, **{f"model_param_{k}": v for k, v in params.items()}}
    result["date"] = str(datetime.datetime.now())

    # Save the data
    if os.path.isfile(TRAINING_RESULTS_FILE):
        df = pd.read_csv(TRAINING_RESULTS_FILE)
        df = df.append(result, ignore_index=True)
    else:
        df = pd.DataFrame.from_records([result])

    df.to_csv(TRAINING_RESULTS_FILE, index=False)


def get_params_from_distributions(params_config: Dict[str, Any]):
    """Iterate over the parameters, retrieving one value from each distribution."""
    params = {}
    for k, v in params_config.items():
        try:
            params[k] = v.rvs(size=1)[0]
        except AttributeError:
            if isinstance(v, list):
                params[k] = choice(v)
            else:
                params[k] = v
    return params


def find_best_model(
    params_config: Dict[str, Any],
    batch_size: int = 1,
    n: int = 10,
):
    """Find the best model with training."""
    # Security check
    if os.path.isfile(TRAINING_RESULTS_FILE):
        logging.warning("We already have a training file, we will not overwrite it")
        raise FileExistsError(TRAINING_RESULTS_FILE)

    # Load the dataloaders
    train_dataset = PriceLocationsDataset(transforms=transforms, dataset="train")
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataset = PriceLocationsDataset(transforms=transforms, dataset="test")
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    for step in range(n):
        params = get_params_from_distributions(params_config)
        logging.info("Starting step %d", step)
        logging.info("Parameters: %s", params)

        # Create the model
        model = get_model(model_type=params.get("model_type", "resnet50"), pretrained=True)
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=params.get("OPTI_LEARNING_RATE"),
            betas=params.get("OPTI_BETA", (0.9, 0.999)),
            weight_decay=params.get("OPTI_WEIGHT_DECAY"),
        )

        # Training the model
        for epoch in range(ceil(params.get("epochs", 10))):
            train_one_epoch(model, optimizer, train_loader, torch.device("cuda:0"), epoch)

        # Evaluating the model
        evaluate_and_save(model, val_loader, torch.device("cuda:0"), params)
