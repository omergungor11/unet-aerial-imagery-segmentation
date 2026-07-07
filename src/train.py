import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PatchSegmentationDataset
from data.mask_utils import CLASS_NAMES
from data.patchify_dataset import build_patch_cache
from models.unet import UNet
from utils.metrics import ConfusionMatrixTracker
from utils.viz import plot_training_curves


def compute_class_weights(masks_path: Path, num_classes: int) -> torch.Tensor:
    masks = np.load(masks_path, mmap_mode="r")
    counts = np.bincount(np.asarray(masks).reshape(-1), minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, num_classes, train: bool):
    model.train() if train else model.eval()
    tracker = ConfusionMatrixTracker(num_classes)
    total_loss = 0.0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks in tqdm(loader, leave=False):
            images, masks = images.to(device), masks.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            tracker.update(preds, masks)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, tracker


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg, model_cfg, train_cfg, augment_cfg = (
        cfg["data"], cfg["model"], cfg["train"], cfg["augment"]
    )

    torch.manual_seed(train_cfg["seed"])
    device = torch.device(train_cfg["device"])

    cache_paths = build_patch_cache(data_cfg)

    train_ds = PatchSegmentationDataset(*cache_paths["train"], augment_cfg=augment_cfg)
    val_ds = PatchSegmentationDataset(*cache_paths["val"], augment_cfg=None)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    model = UNet(**model_cfg).to(device)

    if train_cfg["use_class_weights"]:
        weights = compute_class_weights(cache_paths["train"][1], model_cfg["num_classes"]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"

    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    best_miou, epochs_without_improvement = -1.0, 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, model_cfg["num_classes"], train=True
        )
        val_loss, val_tracker = run_epoch(
            model, val_loader, criterion, optimizer, device, model_cfg["num_classes"], train=False
        )
        val_miou = val_tracker.mean_iou()
        scheduler.step(val_miou)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)

        print(
            f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_mIoU {val_miou:.4f}"
        )

        if val_miou > best_miou:
            best_miou, epochs_without_improvement = val_miou, 0
            torch.save(
                {"model_state": model.state_dict(), "model_cfg": model_cfg, "epoch": epoch, "val_miou": val_miou},
                best_path,
            )
            print(f"  -> yeni en iyi model kaydedildi ({best_path})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["patience"]:
                print(f"Early stopping: {train_cfg['patience']} epoch boyunca val_mIoU iyileşmedi.")
                break

    log_dir = Path(train_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, log_dir / "training_curves.png")
    print(f"Eğitim tamamlandı. En iyi val mIoU: {best_miou:.4f}")
    print(f"Sınıflar: {CLASS_NAMES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
