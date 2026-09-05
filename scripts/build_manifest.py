#!/usr/bin/env python3
r"""
Streamlines image downloading, inspection, hashing, probing, metatadata extraction and saving functionalities (Whihc are performed by the other modules in this project; i.e.- `integrity.py`, `manifest.py`, `selection.py`). This script allows for the same functionalities to be performed across various image sources.

Usage examples:
```
    # GenImage AI images from one generator
    python scripts/build_manifest.py \
        --root data/raw \ 
        --scan data/raw/genimage/stable_diffusion_v_1_4/ai \
        --source genimage --label 1 --generator stable_diffusion_v_1_4 \
        --out data/interim/manifest_sd14.parquet

    # COCO real images
    python scripts/build_manifest.py \
        --root data/raw --scan data/raw/coco/val2017 \
        --source coco --label 0 --generator real --split test_ood_real \
        --out data/interim/manifest_coco.parquet
```


For GenImage images, run the script per generator:
    - stable_diffusion_v_1_4
    - stable_diffusion_v_1_5
    - glide
    - adm
    - vqdm
    - midjourney
    - wukong
    - biggan

Where under each generator, specify the path for storing the AI-generated image (`ai`, label 1) and its real (`nature`, label 0) counterparts; i.e.:
```
# --- GenImage AI images from Stable Diffusion ---

python scripts/build_manifest.py \
    --root data/raw \
    --scan data/raw/genimage/stable_diffusion_v_1_4/ai \
    --source genimage --label 1 --generator stable_diffusion_v_1_4 \
    --content-class-from-parent \
    --out data/interim/manifest_sd14_ai.parquet

# --- Real images paired with Stable Diffusion's AI-generated images ---

python scripts/build_manifest.py \
    --root data/raw \
    --scan data/raw/genimage/stable_diffusion_v_1_4/ai \
    --source genimage --label 0 --generator stable_diffusion_v_1_4 \
    --content-class-from-parent \
    --out data/interim/manifest_sd14_ai.parquet
```

Do this for each generator.

Upon running the script we conceputally have a metadata database such as:
    ======================== ========== ====== ======================== ============== =========== ========== ====== ====== =======
    path                     source     label  generator                content_class  split       sha256     width  height corrupt
    ======================== ========== ====== ======================== ============== =========== ========== ====== ====== =======
    genimage/.../image1.jpg  genimage   1      stable_diffusion_v_1_4   dog            unassigned  a94...     512    512    False
    genimage/.../image2.jpg  genimage   1      stable_diffusion_v_1_4   dog            unassigned  f21...     512    512    False
    ======================== ========== ====== ======================== ============== =========== ========== ====== ====== =======
Where column definitions are determined by `manifest.py`. The above metadat database is saved as a `.parquet` file rather that a `pandas.DataFrame` as `.parquet` is more analytically efficicent.

The databased for each generator are then later combined in an analysis notebook: 

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

def main() -> int:
    ap= argparse.ArgumentParser(description=__doc__, 
                                formatter_class= argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type= Path, required= True, 
                    help= "Path prefix that the manifest paths are relative to")
    ap.add_argument("--scan", type= Path, required= True, 
                    help= "Directory the program will go through (must be under --root)")
    ap.add_argument("--source", required= True, help= "Specific image source: gemimage | ntire| coco | raise")
    ap.add_argument("--label", type= int, required= True, choices= [0, 1], 
                    help= "Image label: 0= Human-made, 1= AI-generated")
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

    paths= discover_images(args.scan)
    if args.limit:
        paths= paths[: args.limit]
    print(f"Found {len(paths)} image files under {args.scan}")
    if not paths:
        return 1

    jobs= [
        (
            p, 
            args.root, 
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
    

