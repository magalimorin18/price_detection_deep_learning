"""Price detection utils."""

from typing import Dict

import pandas as pd
import torch


def convert_model_output_to_format(model_output: Dict) -> pd.DataFrame:
    """Convert the model output to the right format."""
    results_list = []
    boxes = model_output["boxes"].detach()
    if "scores" in model_output:
        scores = model_output["scores"].detach()
    else:
        scores = torch.zeros((boxes.shape[0],))
    for box, score in zip(boxes, scores):
        result = {**dict(zip(["x1", "y1", "x2", "y2"], box.tolist())), "score": score.item()}
        results_list.append(result)

    return pd.DataFrame(results_list)
