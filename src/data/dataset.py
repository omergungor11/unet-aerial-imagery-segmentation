import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchSegmentationDataset(Dataset):
    """Cache'lenmiş (.npy) image/mask patch'lerini okuyan torch Dataset'i."""

    def __init__(self, images_path: Path, masks_path: Path, augment_cfg: dict | None = None):
        self.images = np.load(images_path, mmap_mode="r")
        self.masks = np.load(masks_path, mmap_mode="r")
        assert len(self.images) == len(self.masks)
        self.augment_cfg = augment_cfg or {}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = np.array(self.images[idx])  # (H, W, 3) uint8
        mask = np.array(self.masks[idx])  # (H, W) uint8

        if self.augment_cfg:
            img, mask = self._augment(img, mask)

        img_t = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
        mask_t = torch.from_numpy(mask.astype(np.int64))
        return img_t, mask_t

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        if random.random() < self.augment_cfg.get("hflip", 0):
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < self.augment_cfg.get("vflip", 0):
            img = np.ascontiguousarray(img[::-1])
            mask = np.ascontiguousarray(mask[::-1])
        if random.random() < self.augment_cfg.get("rotate90", 0):
            k = random.choice([1, 2, 3])
            img = np.ascontiguousarray(np.rot90(img, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
        return img, mask
