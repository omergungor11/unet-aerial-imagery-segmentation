import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PatchSegmentationDataset
from data.mask_utils import CLASS_NAMES
from data.patchify_dataset import build_patch_cache
from models.unet import UNet
from utils.metrics import ConfusionMatrixTracker
from utils.viz import plot_confusion_matrix


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]
    device = torch.device(train_cfg["device"])

    cache_paths = build_patch_cache(data_cfg)
    test_ds = PatchSegmentationDataset(*cache_paths["test"], augment_cfg=None)
    test_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet(**checkpoint["model_cfg"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tracker = ConfusionMatrixTracker(model_cfg["num_classes"])
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="test seti değerlendirme"):
            images, masks = images.to(device), masks.to(device)
            preds = model(images).argmax(dim=1)
            tracker.update(preds, masks)

    ious = tracker.per_class_iou()
    dices = tracker.per_class_dice()

    print(f"\n{'Sınıf':<24} {'IoU':>8} {'Dice':>8}")
    for name, iou, dice in zip(CLASS_NAMES, ious, dices):
        print(f"{name:<24} {iou:>8.4f} {dice:>8.4f}")
    print(f"\nMean IoU: {tracker.mean_iou():.4f}")
    print(f"Pixel Accuracy: {tracker.pixel_accuracy():.4f}")

    out_dir = Path(train_cfg["log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(tracker.matrix, CLASS_NAMES, out_dir / "confusion_matrix.png")
    print(f"Confusion matrix kaydedildi -> {out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
