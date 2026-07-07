"""Ham (images/masks) tile verisinden sabit boyutlu patch cache'i üretir.

Instructor'ın load_dataset fonksiyonundan farkı:
- Görüntüleri resize ile bozmak yerine, patch'lere bölerek gerçek çözünürlükte
  ve sabit boyutta veri üretir (görüntüler 509x544 ile 2149x1479 arası
  değişken boyutlarda olduğu için resize ciddi distorsiyona yol açar).
- Maskeler RGB renk kodlu olduğu için grayscale okumak yerine
  mask_utils.rgb_to_class_mask ile class-index maskeye çevrilir.
- Sonuç diske .npy olarak cache'lenir; aynı config ile tekrar çalıştırıldığında
  yeniden hesaplanmaz.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .mask_utils import UNLABELED_CLASS_INDEX, rgb_to_class_mask


def _extract_patches(array: np.ndarray, patch_size: int) -> np.ndarray:
    """(H, W[, C]) dizisini (N, patch_size, patch_size[, C]) patch'lere böler.

    Kalan kenar (patch_size'a tam bölünmeyen kısım) atılır.
    """
    h, w = array.shape[:2]
    n_rows, n_cols = h // patch_size, w // patch_size
    cropped = array[: n_rows * patch_size, : n_cols * patch_size]

    if cropped.ndim == 3:
        c = cropped.shape[2]
        patches = cropped.reshape(n_rows, patch_size, n_cols, patch_size, c)
        patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, c)
    else:
        patches = cropped.reshape(n_rows, patch_size, n_cols, patch_size)
        patches = patches.transpose(0, 2, 1, 3).reshape(-1, patch_size, patch_size)

    return patches


def _config_hash(data_cfg: dict) -> str:
    payload = json.dumps(data_cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def _load_tile(root_dir: Path, tile_id: int, patch_size: int):
    tile_dir = root_dir / f"Tile {tile_id}"
    img_dir, mask_dir = tile_dir / "images", tile_dir / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Tile {tile_id} bulunamadı: {img_dir} (data.root_dir proje kök dizinine göre "
            "doğru mu, script proje kökünden mi çalıştırılıyor kontrol edin)."
        )

    img_patches, mask_patches = [], []
    for img_path in sorted(img_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            continue

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2RGB)
        class_mask = rgb_to_class_mask(mask_rgb)

        img_patches.append(_extract_patches(img, patch_size))
        mask_patches.append(_extract_patches(class_mask, patch_size))

    if not img_patches:
        return np.empty((0, patch_size, patch_size, 3), dtype=np.uint8), np.empty(
            (0, patch_size, patch_size), dtype=np.uint8
        )

    return np.concatenate(img_patches, axis=0), np.concatenate(mask_patches, axis=0)


def _filter_mostly_unlabeled(images: np.ndarray, masks: np.ndarray, max_ratio: float):
    if len(masks) == 0:
        return images, masks
    unlabeled_ratio = (masks == UNLABELED_CLASS_INDEX).mean(axis=(1, 2))
    keep = unlabeled_ratio <= max_ratio
    return images[keep], masks[keep]


def build_patch_cache(data_cfg: dict, force: bool = False) -> dict[str, tuple[Path, Path]]:
    """Config'e göre train/val/test patch cache'ini oluşturur (veya var olanı kullanır).

    Returns: split adı -> (images_npy_path, masks_npy_path)
    """
    root_dir = Path(data_cfg["root_dir"])
    cache_dir = Path(data_cfg["cache_dir"])
    patch_size = data_cfg["patch_size"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = _config_hash(data_cfg)
    paths: dict[str, tuple[Path, Path]] = {}

    for split, tile_ids in data_cfg["splits"].items():
        images_path = cache_dir / f"{split}_images_{cache_key}.npy"
        masks_path = cache_dir / f"{split}_masks_{cache_key}.npy"
        paths[split] = (images_path, masks_path)

        if images_path.exists() and masks_path.exists() and not force:
            continue

        split_images, split_masks = [], []
        for tile_id in tqdm(tile_ids, desc=f"[{split}] patch çıkarma"):
            imgs, masks = _load_tile(root_dir, tile_id, patch_size)
            split_images.append(imgs)
            split_masks.append(masks)

        images = np.concatenate(split_images, axis=0)
        masks = np.concatenate(split_masks, axis=0)

        if data_cfg.get("drop_mostly_unlabeled", False):
            images, masks = _filter_mostly_unlabeled(
                images, masks, data_cfg.get("max_unlabeled_ratio", 1.0)
            )

        np.save(images_path, images)
        np.save(masks_path, masks)
        print(f"[{split}] {len(images)} patch kaydedildi -> {images_path.name}")

    return paths
