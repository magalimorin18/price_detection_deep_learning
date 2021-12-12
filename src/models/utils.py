"""Utils for models."""

import torch
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train one epoch."""
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
