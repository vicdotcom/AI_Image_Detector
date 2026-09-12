"""
Uniform Image Preprocessing
---------------------------

This module actually resizes and re-encodes every surviving image to one fixed geometry and one fixed JPEG quality factor, regardless of class or source, thereby eliminating/minimzing any shortcut signal a machine learning model may learn from these features.

### Data flow

```
    manifest row (raw path, split, label)
            |
            v
    resize_and_reencode()   -- pure function, PIL.Image in -> PIL.Image out
            |
            v
    process_one()           -- adds disk I/O: open raw file, save processed file
            |
            v
    build_processed_dataset() -- fans process_one() out across a process pool
            |
            v
    data/processed/<split>/<ai|human>/<sha256>.jpg
            |
            v
    updated manifest (data/processed/manifest.parquet) ready for a
    torch.utils.data.Dataset in the training step.
```

Why re-encode at all, given the images already went through bias matching?
Matching only filters *which* files survive; it cannot make surviving files bit-identical in compression, because it never touches pixels. Re-encoding every image (both classes, all sources) at the same JPEG quality factor and the same canvas size removes `jpeg_qf`, `width`, and `height` as usable shortcut features (see notebooks/metadata_EDA.ipynb, section 8) without touching the deeper synthesis artifacts a CNN is meant to key on.
"""


from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False  # same safety setting as integrity.py
Image.MAX_IMAGE_PIXELS = 250_000_000


## ==================================================================================
## Configuration
## ==================================================================================
@dataclass(frozen= True)
class PreprocessConfig:
    """
    :param image_size: side length (pixels) of the final square output. The pipeline resizes the shorter side to this value, then center-crops.
    :param jpeg_qf: JPEG quality factor (1-100) every output image is saved at, regardless of its original format or quality.
    :param resample: PIL resampling filter used for the resize step. It specifies how the new pixels should be calculated when the image size is changed. LANCZOS is the default and most optimal resampling method: it preserves high-frequency detail better than bilinear/box filters, which matters because forensic signal often concentrates in high frequencies.
    """

    image_size: int = 512
    jpeg_qf: int = 98
    resample: int = Image.Resampling.LANCZOS

    @classmethod
    def from_yaml_dict(cls, raw:dict) -> PreprocessConfig:
        """
        Builds a `PreprocessConfig` from the `preprocessing:` block of `subset_v1.yaml`.
        
        """
        return cls(
            image_size=int(raw.get("image_size", 512)),
            jpeg_qf=int(raw.get("jpeg_qf", 98)),
        )


## ==================================================================================
## Pure pixel transform
## ==================================================================================
def resize_and_reencode(im: Image.Image, cfg: PreprocessConfig)-> Image.Image:
    """
    Resizes images to a fixed square. It first resizes the shorter side first then center-crops to a fixed square. This is done *after* forcing the images to RGB.

    Data flow is as follows:
    ```
        PIL.Image (any mode, any size)
              |
              v  .convert("RGB")       -- strips alpha/palette/CMYK so every
              |                           output has exactly 3 channels
              v  resize shorter side -> cfg.image_size, preserving aspect ratio
              |
              v  center-crop to (cfg.image_size, cfg.image_size)
              v
        PIL.Image, mode "RGB", size (cfg.image_size, cfg.image_size)
    ```

    We first convert to RGB since we may have images that are in various Pillow image modes including but not limited to:
        - `RGB`- 3x8 bit pixels. True color. The most common format for standard images and web graphics (Red, Green, Blue).
        - `RGBA`- 4x8 bit pixels. True color with an alpha (transparency) mask. Essential for PNGs with see-through backgrounds.
        - `L`- 8-bit pixels, grayscale. Standard black-and-white images with 256 shades of gray.
        - `P`- 80-bit pixels mapped to any other mode using a color palette (indexed color). Often used in GIFs to reduce file size.
    Image modes such as `RGBA`, `L`, `P`, and so on could break the [3, H, W] tensor shape assumed in every downstream model.

    We also perform a shorter-side-first cropping strategy which maintains the image as it was originally.
    """
    im = im.convert("RGB")
    w, h = im.size
    short_side = min(w, h)
    scale = cfg.image_size / short_side
    new_w, new_h = max(cfg.image_size, round(w * scale)), max(cfg.image_size, round(h * scale))
    im = im.resize((new_w, new_h), resample=cfg.resample)

    left = (new_w - cfg.image_size) // 2
    top = (new_h - cfg.image_size) // 2
    im = im.crop((left, top, left + cfg.image_size, top + cfg.image_size))
    return im

## ==================================================================================
## Single-image disk operation
## ==================================================================================
def process_one(job: tuple[Path, Path, PreprocessConfig]) -> dict:
    """
    Opens one raw image, transforms it, saves the result, and reports what happened. Never raises an error. Failures are captured in the returned dict so one bad file can't crash a multi-hour batch job.
    
    Params:
        job (tuple[Path, Path, PreprocessConfig]): ``(src_abs_path, dst_abs_path, cfg)`` tuple. Packed into a single tuple (rather than three positional args) because `ProcessPoolExecutor.map` needs a single iterable of picklable arguments per call, and PreprocessConfig is a small frozen dataclass so it pickles cheaply.

    Returns:
        dict: Contains processed_path / processed_width / processed_height / processed_ok / processed_error. Designed to be assembled into a DataFrame and concatenated onto the input manifest column-wise.
    """
    src, dst, cfg = job
    try:
        with Image.open(src) as im:
            out = resize_and_reencode(im, cfg)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, format="JPEG", quality=cfg.jpeg_qf)
        return {
            "processed_path": str(dst),
            "processed_width": out.width,
            "processed_height": out.height,
            "processed_ok": True,
            "processed_error": None,
        }
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure
        # here must not crash the batch; it must be recorded and inspected.
        return {
            "processed_path": None,
            "processed_width": None,
            "processed_height": None,
            "processed_ok": False,
            "processed_error": f"{type(exc).__name__}: {exc}",
        }


## ==================================================================================
## Batch driver
## ==================================================================================
def build_processed_dataset(
    df: pd.DataFrame,
    raw_root: Path,
    processed_root: Path,
    cfg: PreprocessConfig,
    n_workers: int = 8,
    chunksize: int = 32,
) -> pd.DataFrame:
    """
    Fans `process_one` out across a process pool for every row of a manifest that already has `path`, `split`, `label`, and `sha256` columns.

    Destination layout:
        processed_root / <split> / <"ai" if label==1 else "human"> / <sha256>.jpg

    Using sha256 as the filename (rather than the original filename) keeps
    names collision-free across sources without inventing a new ID scheme,
    and makes a partial rerun idempotent: re-running against a directory
    that already has some of these files just overwrites them with
    byte-identical output (same seed, same deterministic pipeline).

    Rows already marked `is_corrupt` should be filtered out by the caller
    before this is called -- this function assumes every row is expected to
    succeed and treats any failure as worth inspecting.
    """
    jobs = []
    for row in df.itertuples(index=False):
        src = raw_root / str(row.path)
        label_dir = "ai" if row.label == 1 else "human"
        dst = processed_root / str(row.split) / label_dir / f"{row.sha256}.jpg"
        jobs.append((src, dst, cfg))

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(process_one, jobs, chunksize=chunksize):
            results.append(res)

    out = df.reset_index(drop=True).copy()
    res_df = pd.DataFrame(results)
    return pd.concat([out, res_df], axis=1)