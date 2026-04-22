"""Build the hero image for the geospatial post.

Layout: 3x3 grid of the same Sentinel-2 tile (Moorea, CC BY-SA 3.0 IGO,
Copernicus Sentinel-2 / CNES via Wikimedia Commons) under different
Albumentations 2.2.0 transforms.

Output: 1200x1200 webp, designed to live on the right side of a Vercel-style
OG card (text on left, image on right).
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "moorea_source.jpg"
OUT_WEBP = HERE / "hero.webp"
OUT_PNG = HERE / "hero.png"

CARD_SIZE = 1200
GUTTER = 14
BG = (32, 18, 11)  # deep navy in BGR (R=11, G=18, B=32 -> #0B1220)
GRID = 3

CELL = (CARD_SIZE - GUTTER * (GRID + 1)) // GRID  # 391


def load_tile() -> np.ndarray:
    img = cv2.imread(str(SRC), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"missing {SRC}")
    h, w = img.shape[:2]
    # center-square crop on the most photogenic part (Moorea lagoon, left half)
    side = min(h, w) // 2
    cy, cx = h // 2, int(w * 0.42)
    y0 = max(0, cy - side // 2)
    x0 = max(0, cx - side // 2)
    crop = img[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (CELL, CELL), interpolation=cv2.INTER_AREA)


def build_transforms() -> list[tuple[str, A.BasicTransform]]:
    return [
        ("original", A.NoOp()),
        ("vflip", A.VerticalFlip(p=1.0)),
        ("rotate", A.Rotate(angle_range=(25, 25), p=1.0)),
        (
            "brightness",
            A.RandomBrightnessContrast(
                brightness_range=(0.25, 0.25),
                contrast_range=(0.0, 0.0),
                p=1.0,
            ),
        ),
        (
            "hue",
            A.HueSaturationValue(
                hue_shift_range=(20, 20),
                sat_shift_range=(0, 0),
                val_shift_range=(0, 0),
                p=1.0,
            ),
        ),
        ("noise", A.GaussNoise(std_range=(0.06, 0.06), p=1.0)),
        (
            "elastic",
            A.ElasticTransform(alpha=120, sigma=10, p=1.0),
        ),
        (
            "blur",
            A.MotionBlur(blur_range=(21, 21), angle_range=(45, 45), p=1.0),
        ),
        (
            "dropout",
            A.CoarseDropout(
                num_holes_range=(8, 8),
                hole_height_range=(0.08, 0.12),
                hole_width_range=(0.08, 0.12),
                fill=0,
                p=1.0,
            ),
        ),
    ]


def main() -> None:
    rng = np.random.default_rng(42)
    tile = load_tile()
    transforms = build_transforms()
    assert len(transforms) == GRID * GRID

    canvas = np.full((CARD_SIZE, CARD_SIZE, 3), BG, dtype=np.uint8)

    for i, (_name, tfm) in enumerate(transforms):
        row, col = divmod(i, GRID)
        # seed each transform deterministically
        np.random.seed(int(rng.integers(0, 1_000_000)))
        out = tfm(image=tile)["image"]
        if out.shape != (CELL, CELL, 3):
            out = cv2.resize(out, (CELL, CELL), interpolation=cv2.INTER_AREA)
        y = GUTTER + row * (CELL + GUTTER)
        x = GUTTER + col * (CELL + GUTTER)
        canvas[y : y + CELL, x : x + CELL] = out

    cv2.imwrite(str(OUT_PNG), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    cv2.imwrite(str(OUT_WEBP), canvas, [cv2.IMWRITE_WEBP_QUALITY, 88])
    print(f"wrote {OUT_PNG} ({OUT_PNG.stat().st_size // 1024} KB)")
    print(f"wrote {OUT_WEBP} ({OUT_WEBP.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
