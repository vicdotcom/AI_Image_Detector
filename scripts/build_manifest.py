#!/usr/bin/env python3
r"""
Streamlines image downloading, inspection, hashing, probing, metatadata extraction and saving functionalities (Which are performed by the other modules in this project; i.e.- `integrity.py`, `manifest.py`, `selection.py`). This script allows for the same functionalities to be performed across various image sources **(currently for GenImage and COCO Images)

There are two ways of pointing this script at data, chosen via `--layout`:

``--layout flat`` (default)
    Scans one directory of images that all share the same label/generator/split,
    which you supply explicitly. Use this for sources like COCO where a single
    folder holds one homogeneous group of images.

``--layout genimage-tree``
    Walks a GenImage-style dataset root in one pass and derives generator, split,
    and label straight from the folder names, so the whole dataset is scanned
    and written to a single manifest in one invocation instead of one run per
    generator/label pair. Expects the layout:
    ```
    <scan>/
    └── <generator>/
        ├── train/
        │   ├── ai/       (label 1)
        │   └── nature/   (label 0)
        └── val/
            ├── ai/
            └── nature/
    ```

Usage examples:
```
    # Tiny GenImage: whole dataset (all generators, splits, labels) in one pass
    python scripts/build_manifest.py \
        --root data/raw \
        --scan data/raw/tiny_genimage \
        --source genimage --layout genimage-tree \
        --out data/interim/manifest_tiny_genimage.parquet

    # COCO real images (flat layout, label supplied explicitly)
    python scripts/build_manifest.py \
        --root data/raw \
        --scan data/raw/coco/val2017 \
        --source coco --label 0 --generator real --split test_ood_real \
        --out data/interim/manifest_coco.parquet
```

Upon running the script we conceputally have a metadata database such as:
```
path	source	label	generator	content_class	split	sha256	width	height	corrupt
genimage/.../image1.jpg	genimage	1	stable_diffusion_v_1_4	dog	unassigned	a94...	512	512	False
genimage/.../image2.jpg	genimage	1	stable_diffusion_v_1_4	dog	unassigned	f21...	512	512	False
```
Where column definitions are determined by `manifest.py`. The above metadat database is saved as a `.parquet` file rather that a `pandas.DataFrame` as `.parquet` is more analytically efficicent.

**Design note:** `--root` and `--scan` are separate on purpose. `--root` entails the full local machine path while `--scan` consists of the relative machine-agnostic path to allow for reproducibility.
"""



from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
    # File path for module packages (stored within \src)

from ai_detector.data.manifest import( # noqa: E402
    build_manifest, discover_images, save_manifest)

## GenImage-tree leaf folder -> label, per data/raw/tiny_genimage's confirmed layout
GENIMAGE_LABEL_DIRS= {"ai": 1, "nature": 0}


def discover_genimage_tree_jobs(scan: Path, root: Path, source: str) -> list[tuple]:
    """
    Walk a GenImage-style dataset root (``<scan>/<generator>/<split>/{ai,nature}``)
    and build one probe job per image, deriving generator/split/label from the
    folder names.
    """
    jobs: list[tuple] = []
    generator_dirs= sorted(d for d in scan.iterdir() if d.is_dir())
    for gen_dir in generator_dirs:
        split_dirs= sorted(d for d in gen_dir.iterdir() if d.is_dir())
        for split_dir in split_dirs:
            for label_name, label in GENIMAGE_LABEL_DIRS.items():
                label_dir= split_dir / label_name
                if not label_dir.is_dir():
                    continue
                paths= discover_images(label_dir)
                jobs.extend(
                    (root, p, source, label, gen_dir.name, None, split_dir.name)
                    for p in paths
                )
    return jobs


def main() -> int:
    ap= argparse.ArgumentParser(description=__doc__,
                                formatter_class= argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type= Path, required= True,
                    help= "Path prefix that the manifest paths are relative to")
    ap.add_argument("--scan", type= Path, required= True,
                    help= "Directory the program will go through (must be under --root)")
    ap.add_argument("--source", required= True, help= "Specific image source: gemimage | ntire| coco | raise")
    ap.add_argument("--layout", choices= ["flat", "genimage-tree"], default= "flat",
                    help= "'flat': one directory, one label/generator/split supplied via the flags below. "
                    "'genimage-tree': scan a whole GenImage-style dataset root in one pass, "
                    "deriving generator/split/label from the <generator>/<split>/{ai,nature} folder names.")
    ap.add_argument("--label", type= int, choices= [0, 1], default= None,
                    help= "Image label: 0= Human-made, 1= AI-generated. Required for --layout flat; "
                    "ignored (derived from folder names) for --layout genimage-tree.")
    ap.add_argument("--generator", default= "unknown", help= "'real' for human-made images, model name for AI images")
    ap.add_argument("--split", default="unassigned") # train/val partition
    ap.add_argument("--content-class-from-parent", action="store_true",
                    help="Use the immediate parent directory name as content_class "
                    "(works for ImageNet-style folder layouts).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Probe only the first N files. Recommended for use when developing the pipeline.")
    ap.add_argument("--workers", type=int, default=None,
                    help= "How many workers will process the image simultaneously")
    ap.add_argument("--out", type=Path, required=True,
                    help= "Where the complete manifest will be stored")
    args = ap.parse_args()

    if args.layout == "flat" and args.label is None:
        ap.error("--label is required when --layout flat")

    if args.layout == "genimage-tree":
        jobs= discover_genimage_tree_jobs(args.scan, args.root, args.source)
        if args.limit:
            jobs= jobs[: args.limit]
        print(f"Found {len(jobs)} image files under {args.scan}")
        if not jobs:
            return 1
    else:
        paths= discover_images(args.scan)
        if args.limit:
            paths= paths[: args.limit]
        print(f"Found {len(paths)} image files under {args.scan}")
        if not paths:
            return 1

        jobs= [
            (
                args.root,
                p,
                args.source,
                args.label,
                args.generator,
                p.parent.name if args.content_class_from_parent else None,
                args.split
            )
            for p in paths
            ]

    df= build_manifest(jobs, n_workers=  args.workers)
    record= save_manifest(df, args.out, note= f"scan of {args.scan}")

    print(f"\nWrote {record['n_rows']} rows -> {args.out}")
    print(f"  corrupt files : {record['counts']['corrupt']}")
    print(f"  unique sha256 : {df['sha256'].nunique()}  (exact duplicates: "
          f"{len(df) - df['sha256'].nunique()})")
    print(f"  manifest sha  : {record['sha256'][:16]}...")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
