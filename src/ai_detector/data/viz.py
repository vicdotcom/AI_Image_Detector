"""
Plot helpers for EDA on metadata table and actual images. The mental model is as follows:

```
                    DATASET
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
       METADATA TABLE        IMAGE FILES
             │                   │
             │                   │
             ▼                   ▼
      Statistical EDA       Visual EDA
             │                   │
       ┌─────┴─────┐       ┌─────┴──────┐
       │           │       │            │
       ▼           ▼       ▼            ▼
 size_scatter  histograms image_grid  pair_grid
       │           │       │            │
       └───────────┴───────┴────────────┘
                       │
                       ▼
              FIND DATASET PROBLEMS
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼               ▼
      bias          leakage        mislabels
        │              │               │
        └──────────────┼───────────────┘
                       ▼
                REVISE DATASET
                       │
                       ▼
                TRAIN DETECTOR
```
"""

from __future__ import annotations
from typing import Sequence

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

## Directory for saving images
FIGDIR= Path("reports/figures")


## Saving Figures ---------------------------------------------------------
def save(fig: plt.Figure, name: str, figdir: Path = FIGDIR) -> Path:
    """
    Saves a figure at a consistent size/DPI (quality) and returns its path.

    Parameters:
        fig (plt.Figure): The generated figure
        name (str): Name of the generated figure
        figdir (Path): The path where the figure will be saved

    Returns:
        Path (Path): File path where the figure is saved
    """
    figdir.mkdir(parents= True, exist_ok= True)
    out= figdir / f"{name}.png"
    fig.savefig(out, dpi= 140, bbox_inches= "tight")
    return out



## EDA Plot 1 (Seperability) -------------------------------------------------------------
def size_scatter(df: pd.DataFrame, hue: str = "generator", max_points: int = 400) -> plt.Figure:
    """
    Width vs. height per generator. This function assesses image dimensions from each generator and checks whether they are seperable. For instance, we may find:
    ```
    human images:
    mostly 600x400

    Generator A:
    mostly 1024x1024

    Generator B:
    mostly 512x512
    ```
    Which is something we do not want inherent in our data as it can introduce bias during model training. 

    Params:
        df (pd.DataFrame): Image dataset
        hue (str): Image generator (e.g.- Stable Diffusion, Midjourney, real image, e.t.c)
        max_points (int): Maximum number of points to plot if the image group has fewer points than `max_points`, the function plots all of the points from the image group
    
    Returns:
        plt.Figure: Scatterplot of width vs. height per generator

    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, grp in df.groupby(hue, observed=True):
        sample = grp.sample(min(len(grp), max_points), random_state=0)
        ax.scatter(sample["width"], sample["height"], s=4, alpha=0.35, label=str(key))
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title("Image dimensions by source")
    ax.legend(markerscale=3, fontsize=8)
    return fig


## EDA Plot 2 -------------------------------------------------------------
def dist_by_class(df: pd.DataFrame, col: str, bins: int = 60, log: bool = False) -> plt.Figure:
    """
    Overlaid histogram of `col` for real vs AI.

    Compare the distribution of a numerical feature between human and AI images. Also checks for seperability between them

 
    The question this answers: could I draw a vertical line that separates the
    two histograms? Any visible separation is label information leaking
    through a channel that has nothing to do with generative artifacts.

    Params:
        df (pd.DataFrame): Image dataset
        col (str): Image dataset column names. They should be numeric columns 
        bins (int): Number of bins for the histogram
        log (bool): `True` for a logarithmic x-axis, otherwise the default axis applies

    Returns:
        plt.Figure: Histogram of density against image feature
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, name in [(0, "human"), (1, "AI")]:
        vals = df.loc[df["label"] == label, col].dropna()
        if len(vals):
            ax.hist(vals, bins=bins, alpha=0.55, label=name, density=True)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(col)
    ax.set_ylabel("density")
    ax.set_title(f"{col} by class")
    ax.legend()
    return fig


## EDA Plot 3 ------------------------------------------------------------
def image_grid(
    paths: Sequence[Path],
    titles: Sequence[str] | None = None,
    ncols: int = 8,
    thumb: int = 160,
        ) -> plt.Figure:
    """
    Thumbnail grid. Conceptually: 
    ```
    ┌────┬────┬────┬────┬────┬────┬────┬────┐
    │img │img │img │img │img │img │img │img │
    ├────┼────┼────┼────┼────┼────┼────┼────┤
    │img │img │img │img │img │img │img │img │
    ├────┼────┼────┼────┼────┼────┼────┼────┤
    │img │img │img │img │img │img │img │img │
    └────┴────┴────┴────┴────┴────┴────┴────┘
    ```
    Statistics tell you about distributions. Only your eyes catch watermarks,
    letterboxing, duplicated subjects and mislabelled files.

    Parameters:
        paths (Sequence[Path]): The image files to display.
        titles (Sequence[str] | None): Optional labels for each image.
        ncols (int): Number of columns.
        thumb (int): Thumbnail size (e.g- 160x160).

    Returns:
        plt.Figure: Plot showing a grid of various images.

    """
    n = len(paths)
    nrows = int(np.ceil(n / ncols)) # Accounts for when rows are incomplete
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.7, nrows * 1.9))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, p, i in zip(axes, paths, range(n)):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb))
                ax.imshow(im)
        except Exception as exc:  # noqa: BLE001
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center", fontsize=7)
        if titles is not None and i < len(titles):
            ax.set_title(str(titles[i]), fontsize=6)
    fig.tight_layout()
    return fig


## EDA Plot 4 ------------------------------------------------------------
def pair_grid(pairs: Sequence[tuple[Path, Path, int]], max_pairs: int = 8) -> plt.Figure:
    """
    Show near-duplicate candidates side by side with their bit distance.
 
    Used to choose the Haming Distance threshold. Walk the distance up until the
    pairs stop looking like the same picture, then step back one. Recommended threshold is ~5.

    Params:
        pairs (Sequence[tuple[Path, Path, int]]): This is a tuple containing: `(image A, image B, Hamming distance)`. 
        max_pairs (int): For managing the number of candidate pairs displayed
    
    Returns:
        plt.Figure: Plot of image pairs side by side with their Hamming Distances.
    """
    pairs = list(pairs)[:max_pairs]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(5, 2.4 * len(pairs)))
    axes = np.atleast_2d(axes)
    for row, (a, b, dist) in enumerate(pairs):
        for col, p in enumerate((a, b)):
            ax = axes[row, col]
            ax.axis("off")
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im.thumbnail((220, 220))
                    ax.imshow(im)
            except Exception:  # Unreadable images
                ax.text(0.5, 0.5, "unreadable", ha="center", va="center")
        axes[row, 0].set_title(f"hamming = {dist}", fontsize=8, loc="left")
    fig.tight_layout()
    return fig