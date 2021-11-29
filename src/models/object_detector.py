"""Object detector."""

import logging
import os
from typing import List

import pandas as pd
import torch


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


if __name__ == "__main__":
    model = ObjectDetector()
    results = model.extract_objects(
        [os.path.join("data", "train", "images", str(i).rjust(4, "0") + ".jpg") for i in range(1)]
    )
    print(results.head(10))
