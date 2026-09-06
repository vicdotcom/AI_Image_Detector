# AI Image Detection

An end-to-end computer vision project that classifies whether an image is
**AI-generated** or **human-made**. Built as a learning project with an emphasis
on avoiding dataset shortcuts (see [Dataset Strategy](#2-dataset-strategy))/.

**Current stage:** Environment, provenance tooling, and metadata EDA are in
place; a mini image set (tiny-GenImage) has been downloaded. Next up is uilding the dataset manifest (image hashing, image processing, image data train/val/test splits) from the downloaded.

## Table of Contents
- [AI Image Detection](#ai-image-detection)
  - [Table of Contents](#table-of-contents)
  - [1. Problem Definition](#1-problem-definition)
  - [2. Dataset Strategy](#2-dataset-strategy)
  - [3. Project Structure](#3-project-structure)
  - [4. Setup](#4-setup)
  - [5. Usage](#5-usage)

## 1. Problem Definition

In a real-world setting there are three classes of images that we may encounter:

- **Fully synthetic (positive class):** text-to-image output — Stable Diffusion,
  Midjourney, DALL·E, etc.
- **Human-authored (negative class):** camera photos, scanned art, hand-made
  digital illustration.
- **AI-edited / hybrid** (in-painting, generative fill, upscaling, style
  transfer): This is out of the scope of this project for now, though robustness to these image types will be evaluated in a later stage.

Formally, this is binary classification problem where: given a pixel tensor
$x \in \mathbb{R}^{H \times W \times 3}$, predict $y \in \{\text{human, AI}\}$
with a continuous score.

This is however not a straight-forward task. A model trained on one generator family tends to learn that family's fingerprint rather than "AI-ness" in general, so accuracy can collapse on an unseen generator (a form of [distribution shift](https://parasdahal.com/notes/distribution-shift/)).
This can be due to the following reasons: 
- Generator developers are actively optimizing to eliminate the very artifacts we are seeking to detect
- real-world post-processing (resizing, recompressing, screenshots, re-upload) erodes forensic signal
- If the two classes differ systematically in resolution/format/compression, etc..., a model will happily learn *that* instead.

We aim to produce the best probabilistic estimate from a model fit to a specific distribution. That is: *image is likely AI-generated (model score 0.91)*. Our objective and scope for the project is therefore as follows: 
> Build a binary image classifier that, given a single still image, outputs a calibrated probability that the image was fully synthesized by a generative model.

> This classifier will be trained on multiple diffusion-family generators and multiple real-image sources, and evaluated primarily on held-out generators and post-processed images rather than on i.i.d. (independent and identically distributed) test accuracy.

## 2. Dataset Strategy

Candidate sources include
- **[GenImage](https://github.com/gendetection/UnbiasedGenImage)** (*Primary image source*) — ~1M
  real (ImageNet) / fake pairs across 8 generators, with deliberate bias
  controls (matched sizes, controlled JPEG compression).
- **[NTIRE Robust AIGen Detection](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train)**
  — 42 generators, unlabelled augmentations. Used later as a true wild/out-ofdistribution test.
- **[COCO](https://cocodataset.org/#overview)** — solely real images, used to assess the false-positive rate on an unseen real-image source.
- **[RAISE](https://loki.disi.unitn.it/RAISE/)** — uncompressed RAW-derived
  images; the hardest real-image shift.

**Splitting philosophy:** naive random splitting leaks information via
near-duplicates, paired real/fake images, and generator- or source-specific
signatures. Splits here will instead be **group-aware** (duplicate clusters
never cross a split), **stratified** by class/generator/category, and include
a **held-out-generator** test set so cross-generator generalization, not just
in-distribution accuracy, gets measured.

Starting scale is intentionally small (~1,000–35,000 images, ~50/50 balance)
to get the pipeline and evaluation trustworthy before scaling up to the full
~500GB GenImage dataset.

## 3. Project Structure

```yaml
ai-image-detector/
├── README.md                          # this file
├── pyproject.toml                     # dependencies, dev tools (ruff, pytest), editable install
├── .gitignore                         # excludes data/, secrets, caches, notebook outputs
│
├── configs/
│   └── data/
│       └── subset_v1.yaml             # dataset selection spec: generators, splits, bias-matching thresholds
│
├── data/                              # gitignored — nothing here is committed
│   ├── raw/                           # immutable downloads (tiny_genimage/, coco/, genimage/, ...)
│   ├── interim/                       # work-in-progress manifests, EDA parquet files
│   └── processed/                     # final train/val/test manifests (not yet produced)
│
├── notebooks/
│   ├── 01_genimage_metadata_eda.ipynb # metadata-only EDA; decides dataset composition before downloading images
│   └── 02_image_level_eda.ipynb       # pixel-level EDA (dimensions, corruption, duplicates) — runs once images + manifest exist
│
├── reports/
│   ├── figures/                       # curated PNGs exported from notebooks (committed)
│   ├── manifests/                     # small provenance/version JSON records (committed)
│   └── data_card.md                   # written record of dataset decisions and their justification
│
├── scripts/                           # runnable entry points (thin CLI wrappers over src/)
│   ├── setup_git.sh                   # one-time repo/branch initialization
│   ├── download_genimage_metadata.py  # fetches the small GenImage metadata CSV (not the images)
│   ├── download_images.py             # downloads actual image data: tiny | coco | genimage | ntire | raise | verify
│   └── build_manifest.py              # builds the dataset manifest from images on disk (next step)
│
├── src/ai_detector/                   # installable package — the reusable logic notebooks/scripts import
│   ├── data/
│   │   ├── integrity.py               # corruption checks, SHA-256, perceptual hashing, JPEG quality estimation
│   │   ├── manifest.py                # manifest schema (ImageRecord) and read/write logic
│   │   ├── selection.py               # applies configs/data/*.yaml: bias matching, stratified sampling
│   │   ├── viz.py                     # reusable EDA plotting functions
│   │   └── download.py                # per-source download handlers used by scripts/download_images.py
│   └── utils/                         # logging, seeding, and other shared helpers
│
└── tests/
    └── test_data_integrity.py         # unit tests for hashing/dedup logic — catches silent data-pipeline bugs
```


## 4. Setup

```bash
git clone <this-repo>
cd Ai_Image_Detector

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -e ".[dev]"          # editable install + pytest/ruff/jupyterlab

bash scripts/setup_git.sh        # one-time: branch setup (read it first)
pytest                           # confirm everything is wired correctly
```

`-e` (editable) means edits under `src/ai_detector/` take effect immediately
without reinstalling — verify with:

```bash
python -c "from ai_detector.data.integrity import phash; print('ok')"
```

## 5. Usage

To reproduce the project's current state from scratch, run these in order:

**Step 1: Download GenImage metadata (not the images yet)**

```bash
python scripts/download_genimage_metadata.py --dest data/raw/genimage_meta
```

Fetches the small metadata CSV describing GenImage's ~1M images (dimensions,
generator, JPEG quality) and writes a `provenance.json` recording the source
URL, SHA-256, and download time. This is what lets you plan a dataset subset
without touching the 500GB of actual images.

**Step 2: Metadata-level EDA**

```bash
jupyter lab notebooks/01_genimage_metadata_eda.ipynb
```

Explores class/generator/size/compression distributions from the metadata
CSV, quantifies dataset shortcuts (e.g. how well a model could classify
real-vs-fake using *only* image dimensions and JPEG quality), and produces
`configs/data/subset_v1.yaml` — the dataset selection spec used downstream.

**Step 3 — Download a mini image set (tiny-GenImage)**

```bash
# One-time Kaggle authentication
kaggle auth login

# Recommended first step: ~8GB, 35,000 images across 7 generators
python scripts/download_images.py tiny
```

This gives a small, pre-sampled slice of GenImage — enough to build and debug
the full pipeline (manifest → EDA → splits → baselines) before committing to
the full multi-hundred-GB dataset. Other sources (`coco`, `genimage`, `ntire`,
`raise`) follow the same CLI pattern; run
`python scripts/download_images.py --help` for details, and
`python scripts/download_images.py verify` to check what's on disk.

Every download handler writes a `provenance.json` alongside the data,
recording what was downloaded, when, and its checksum — commit these files
(they're small); actual images (`data/`) are gitignored.