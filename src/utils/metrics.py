import numpy as np
import torch


class ConfusionMatrixTracker:
    """Epoch/split boyunca confusion matrix biriktirip IoU/Dice hesaplar."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().cpu().numpy().reshape(-1)
        targets = targets.detach().cpu().numpy().reshape(-1)
        n = self.num_classes
        idx = n * targets + preds
        cm = np.bincount(idx, minlength=n * n).reshape(n, n)
        self.matrix += cm

    def reset(self) -> None:
        self.matrix[:] = 0

    def per_class_iou(self) -> np.ndarray:
        cm = self.matrix
        intersection = np.diag(cm)
        union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, intersection / union, np.nan)
        return iou

    def per_class_dice(self) -> np.ndarray:
        cm = self.matrix
        intersection = np.diag(cm)
        denom = cm.sum(axis=0) + cm.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dice = np.where(denom > 0, 2 * intersection / denom, np.nan)
        return dice

    def mean_iou(self) -> float:
        return float(np.nanmean(self.per_class_iou()))

    def pixel_accuracy(self) -> float:
        cm = self.matrix
        return float(np.diag(cm).sum() / max(cm.sum(), 1))
