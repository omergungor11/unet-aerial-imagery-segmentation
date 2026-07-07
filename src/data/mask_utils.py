"""RGB renk-kodlu maskeler ile class-index maskeler arasında dönüşüm.

Not: classes.json dosyasındaki hex renkler, gerçek mask PNG'lerindeki piksel
değerleriyle eşleşmiyor (bu veri setinde bilinen bir tutarsızlık). Aşağıdaki
COLOR2CLASS eşlemesi, mask dosyaları üzerinde ölçülen gerçek RGB değerlerine
göre kuruldu.
"""

import numpy as np

CLASS_NAMES = [
    "Building",
    "Land (unpaved area)",
    "Road",
    "Vegetation",
    "Water",
    "Unlabeled",
]

UNLABELED_CLASS_INDEX = CLASS_NAMES.index("Unlabeled")

# class index -> (R, G, B), gerçek mask piksel renkleri
CLASS_COLORS = {
    0: (60, 16, 152),    # Building   #3C1098
    1: (132, 41, 246),   # Land       #8429F6
    2: (110, 193, 228),  # Road       #6EC1E4
    3: (254, 221, 58),   # Vegetation #FEDD3A
    4: (226, 169, 41),   # Water      #E2A929
    5: (155, 155, 155),  # Unlabeled  #9B9B9B
}


def rgb_to_class_mask(mask_rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB mask -> (H, W) uint8 class-index mask."""
    class_mask = np.full(mask_rgb.shape[:2], UNLABELED_CLASS_INDEX, dtype=np.uint8)
    for class_idx, color in CLASS_COLORS.items():
        matches = np.all(mask_rgb == np.array(color, dtype=mask_rgb.dtype), axis=-1)
        class_mask[matches] = class_idx
    return class_mask


def class_mask_to_rgb(class_mask: np.ndarray) -> np.ndarray:
    """(H, W) class-index mask -> (H, W, 3) RGB, görselleştirme için."""
    rgb = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for class_idx, color in CLASS_COLORS.items():
        rgb[class_mask == class_idx] = color
    return rgb
