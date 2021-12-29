"""Utils for models."""

import datetime
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import TRAINING_RESULTS_FILE


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
    model.train()
    losses = {
        k: [] for k in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    }
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluation", total=len(data_loader)):
            # Put the data on the device
            images = list(image.to(device) for image in images)

            targets = [{k: v.squeeze(0).to(device) for k, v in targets.items()}]

            # Pass on the model + Compute the loss
            loss_dict = model(images, targets)
            for k, v in loss_dict.items():
                losses[k].append(v.item())
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


def iterate_params(params_config):
    """Iterate over the parameters, retrieving one parameter from each distribution."""
    for key, value in params_config.items():
        if isinstance(value, dict):
            for k, v in value.items():
                yield (key, k, v)
        else:
            yield (key, None, value)


def find_best_model(model, train_loader, val_loader, device, params_config, n: int = 10):
    """Find the best model with training."""
    params = dict()
    # TODO: Write the code to find the best model
