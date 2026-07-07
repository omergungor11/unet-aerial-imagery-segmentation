import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from data.mask_utils import class_mask_to_rgb
from models.unet import UNet
from utils.viz import overlay_prediction


def _pad_to_multiple(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded, (h, w)


@torch.no_grad()
def predict_image(model, image_rgb: np.ndarray, patch_size: int, device) -> np.ndarray:
    """Görüntüyü patch'lere böler, her patch'i tahmin eder, birleştirir (stitch)."""
    padded, (orig_h, orig_w) = _pad_to_multiple(image_rgb, patch_size)
    ph, pw = padded.shape[:2]
    pred_mask = np.zeros((ph, pw), dtype=np.uint8)

    for y in range(0, ph, patch_size):
        for x in range(0, pw, patch_size):
            patch = padded[y : y + patch_size, x : x + patch_size]
            tensor = torch.from_numpy(patch.transpose(2, 0, 1).astype(np.float32) / 255.0)
            tensor = tensor.unsqueeze(0).to(device)
            logits = model(tensor)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            pred_mask[y : y + patch_size, x : x + patch_size] = pred

    return pred_mask[:orig_h, :orig_w]


def main(config_path: str, checkpoint_path: str, image_path: str, out_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["train"]["device"])
    patch_size = cfg["data"]["patch_size"]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet(**checkpoint["model_cfg"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    pred_mask = predict_image(model, image_rgb, patch_size, device)
    pred_rgb = class_mask_to_rgb(pred_mask)
    overlay = overlay_prediction(image_rgb, pred_rgb, alpha=0.5)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Tahmin overlay kaydedildi -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/predictions/overlay.png")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.image, args.out)
