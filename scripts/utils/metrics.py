from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def _to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _squeeze_binary_map(array: np.ndarray) -> np.ndarray:
    # Accept [B,1,H,W], [B,H,W], [1,H,W], [H,W].
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return array


def calculate_metrics(predictions, targets, threshold: float = 0.5) -> Dict[str, float]:
    predictions = _squeeze_binary_map(_to_numpy(predictions))
    targets = _squeeze_binary_map(_to_numpy(targets))

    predictions = (predictions > threshold).astype("uint8")
    targets = (targets > threshold).astype("uint8")

    flat_preds = predictions.reshape(-1)
    flat_targets = targets.reshape(-1)

    f1 = f1_score(flat_targets, flat_preds, zero_division=0)
    precision = precision_score(flat_targets, flat_preds, zero_division=0)
    recall = recall_score(flat_targets, flat_preds, zero_division=0)

    intersection = np.logical_and(flat_preds, flat_targets).sum()
    union = np.logical_or(flat_preds, flat_targets).sum()
    miou = float(intersection / union) if union > 0 else 1.0

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "miou": float(miou),
    }


def batch_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    preds = torch.sigmoid(predictions)
    return calculate_metrics(preds, targets, threshold=threshold)
