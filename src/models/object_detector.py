"""Object detector."""

import logging
import os
from typing import List

import pandas as pd
import torch
from tqdm import tqdm


# pylint: disable=too-few-public-methods
class ObjectDetector:
    """Object detector using YOLO."""

    def __init__(
        self, model_name: str = "ultralytics/yolov5", model_version: str = "yolov5s"
    ) -> None:
        """Init."""
        self.model = torch.hub.load(repo_or_dir=model_name, model=model_version, pretrained=True)
        logging.info(f"[Object Detector] Now ready to operate!")

    def extract_objects(self, images: List[str]) -> pd.DataFrame:
        """Extract objects.

        :param images: A list of images pathes.

        :returns: All the boxes (as dataframe),
        columns: img_name,x1,x2,y1,y2,yolo_confidence,yolo_class_id,yolo_class_name.
        """
        logging.info("[Object Detector] Prediction on %i images", len(images))
        results = self.model(images[:])
        final_results = []
        for image_path, result in zip(images, results.pandas().xyxy):
            result["img_name"] = os.path.basename(image_path)
            result = result.rename(
                columns={
                    "xmin": "x1",
                    "ymin": "y1",
                    "xmax": "x2",
                    "ymax": "y2",
                    "confidence": "yolo_confidence",
                    "class": "yolo_class_id",
                    "name": "yolo_class_name",
                }
            )

            final_results.append(result)
        final_df: pd.DataFrame = pd.concat(final_results, ignore_index=True)

        # Convert float to int for coordinates
        to_change_cols = ["x1", "x2", "y1", "y2"]
        for col in to_change_cols:
            final_df[col] = final_df[col].astype(int)

        return final_df


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train one epoch."""
    model.train()
    optimizer.zero_grad()

    # WORK ONLY FOR ONE ELEMENT PER BATCH FOR NOW
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}", total=len(data_loader)):
        # Put the data on the device
        images = list(image.to(device) for image in images)

        targets = [{k: v.squeeze(0).to(device) for k, v in targets.items()}]

        # Pass on the model + Compute the loss
        optimizer.zero_grad()
        loss_dict = model(images, targets)

        losses: torch.Tensor = sum(loss for loss in loss_dict.values())
        print(losses.item())

        # Compute gradient and optimize
        losses.backward()
        optimizer.step()
    return losses
