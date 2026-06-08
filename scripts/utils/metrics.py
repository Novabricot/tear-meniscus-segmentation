from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def _to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def calculate_metrics(predictions, targets, threshold: float = 0.5) -> Dict[str, float]:
    predictions = _to_numpy(predictions)
    targets = _to_numpy(targets)

    if predictions.ndim == 4 and predictions.shape[1] == 1:
        predictions = predictions[:, 0, :, :]
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0, :, :]

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
    preds = torch.sigmoid(predictions) if predictions.ndim == 4 else predictions
    return calculate_metrics(preds, targets, threshold=threshold)
