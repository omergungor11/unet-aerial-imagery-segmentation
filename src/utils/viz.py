from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def overlay_prediction(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """image (H,W,3) uint8 + mask_rgb (H,W,3) uint8 -> alpha-blend overlay."""
    return (image.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha).astype(np.uint8)


def plot_training_curves(history: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["val_miou"], label="val mIoU")
    axes[1].set_title("Mean IoU")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
