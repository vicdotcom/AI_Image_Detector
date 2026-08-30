"""
This module turns raw metadata table into a leakage-safe set of splits. 

The key idea behind this module is to control for any experimental bias where the AI image detector may "cheat" its way to accuracy by simply because the AI-generated images happen to have different JPEG quality, dimensions, generators, or duplicated content than human-made images.

It relies on the image checks (i.e.- hashing, decodability, JPEG quality) computed using `integrity.py` and the sbsequent metadata manifest created by `manifest.py`.

```
Raw image collection
       │
       ▼
   integrity.py
       │
       ▼
Check files / hashes / grouping
       │
       ▼
   manifest.py
       │
       ▼
Create metadata manifest
       │
       ▼
  selection.py
       │
       ├── Normalize column names
       │
       ├── Match real-image metadata to fake-image metadata
       │
       ├── Measure metadata shortcuts
       │
       ├── Prevent train/test leakage
       │
       ├── Create train / validation / test / OOD
       │
       └── Optionally create a small pilot dataset
       │
       ▼
Final experimental dataset
```
"""

from __future__ import annotations
from typing import Any # Allows type hinting into Any datatype

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

import yaml
  # The experiment parameters are written directly into a yaml file


## ==================================================================================
## Configuration
## ==================================================================================
@dataclass
class SubsetConfig:
    """
    Custom data structure for the dataset's and model's configuration parameters. Includes:
      - In distribution and out of distribution generators
      - Image matching parameters to ensure consistency across image dimensions and statistics
      - validation and in-distribution test set splits
      - Optional pilot dataset construction parameters
    """
    name: str # Name of the current config (e.g.- baseline, pilot, strict matching)
    seed: int = 42
    real_generator_token: str = "nature" 
      # We are sourcing the human-made images from GenImage. This is how they label them
    train_generators: list[str] = field(default_factory= list)
      # The AI generators implemented in the training data
      # field(default_factory= list) creates a brand new empty list for each class SubsetConfig object 
    ood_generators: list[str] = field(default_factory= list)
      # AI generators not part of the training set that are used for testing/evaluation
   
    # Image matching parameters (matching)
    min_side: int | None = None
    max_side: int | None = None
    jpeg_qf: int | None = None
    jpeg_qf_tolerance: int = 0 # Allows a small range around the JPEG QF value

    # Split fractions (Fractions of image groups rather than for individual images) (splits)
    val_fraction: float = 0.1 # 10% of groups becomes part of the validation set
    test_in_dist_fraction: float = 0.1 # 10% of groups becomes part of the in-distribution test set

    # (pilot)
    pilot_n_per_stratum: int | None = None
      # Allows us to construct a small version of the dataset
      # e.g.- `pilot_n_per_stratum = 50` means 50 images per generator/label/split combination

    
    # Constructing an object from a yaml file 
    @classmethod
    def from_yaml(cls, path: Path) -> SubsetConfig:
        """
        This method constructs the object from reading a yaml file in the `path` specified.
        """
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) 
          # where .safe_load() converts YAML into Python objects
          # This operation should return a dictionary of the SubsetConfig attributes and their values/elements for a particular SubsetConfig object

        # Extracting configuration sections
        matching = raw.pop("matching", {}) or {}
        splits = raw.pop("splits", {}) or {}
        pilot = raw.pop("pilot", {}) or {}
          # If the yaml contains these sections, their data will be extracted from the `raw` dictionary via `.pop()`. Otherwise an empty dictionary will be returned if the config section or data within the section is unavailable 

        merged= {** raw, # Populate this dictionary with the data obtained from `raw`
                 "min_side": matching.get("min_side"), 
                   # .get() returns the value for a dictionary key
                 "max_side": matching.get("max_side"),
                 "jpeg_qf": matching.get("jpeg_qf"),
                 "jpeg_qf_tolerance": matching.get("jpeg_qf_tolerance", 0),
                 "val_fraction": splits.get("val_fraction", 0.1),
                 "test_in_dist_fraction": splits.get("test_in_dist_fraction", 0.1),
                 "pilot_n_per_stratum": pilot.get("n_per_stratum"),
                }

        # The names of the fields declared in SubsetConfig class. This ensures only recognized fields are returned
        known= {f.name for f in cls.__dataclass_fields__.values()} 
        return cls(**{k: v for k, v in merged.items() if k in known})


## ==================================================================================
## Column Normalization
## ==================================================================================
CANONICAL_COLUMNS = {
    # canonical name -> candidate names seen in the wild as we may be importing images from various sources
    "generator": ["generator", "model", "source_model"],
    "width": ["width", "w", "img_width"],
    "height": ["height", "h", "img_height"],
    "jpeg_qf": ["compression_rate", "jpeg_qf", "quality", "qf", "quality_factor"],
    "content_class": ["content_class", "class_id", "class", "label_id", "wnid"],
    "path": ["path", "filepath", "file", "filename", "image_path"],
    "split": ["split", "subset", "partition"],
}

def normalize_columns(df: pd.DataFrame, extra: dict[str, str] | None =  None) -> pd.DataFrame:
    """
    Renaming source-specific columns into consistent canonical names. 

    It receives a `DataFrame` and optionally, a manually specified mapping (`extra`) of `dict` type. 

    To manually define mappings add to the `extra` parameter a dictionary with the column name to be changed as key and the intended column name as value for instance:
    ```python
    extra = {"model_generator":"generator"}
    ```
    
    If no `extra` mappings are specified, the function will perform a default mapping for any of the below cases found:
    ```python
    CANONICAL_COLUMNS = {
    # canonical name -> candidate names seen in the wild as we may be importing images from various sources
    "generator": ["generator", "model", "source_model"],
    "width": ["width", "w", "img_width"],
    "height": ["height", "h", "img_height"],
    "jpeg_qf": ["compression_rate", "jpeg_qf", "quality", "qf", "quality_factor"],
    "content_class": ["content_class", "class_id", "class", "label_id", "wnid"],
    "path": ["path", "filepath", "file", "filename", "image_path"],
    "split": ["split", "subset", "partition"],
    }
    ```

    """

    mapping: dict[str, str] = {}
    lowered= {c.lower(): c for c in df.columns}
      # this becomes a dictionary with the lowercase name as key and the original name as a value (e.g.- "width": "WIDTH")
    for canonical, candidates in CANONICAL_COLUMNS.items(): # For every canonical name.....
        for cand in candidates: # .....It checks the candidate names....
            if cand in lowered and lowered[cand] != canonical: 
                # ....and if the candidate name is found but is not a canonical name.....
                mapping[lowered[cand]] = canonical 
                  # ....the candidate name is renamed to fit the canonical name
                  # Returns something like: "img_height":"height"
                break
    if extra:
        mapping.update(extra) 
          # This occurs after the above canpnical mapping so whatever is specified in extra takes precedence
    return df.rename(columns= mapping) 
      # Canonical names are applied to the image metadata dataset



## ==================================================================================
## Bias Matching
## ==================================================================================
def apply_matching(df: pd.DataFrame, cfg: SubsetConfig)-> pd.DataFrame:
    r"""
    This function restricts real images so their metadata occupies approximately the same region as the generated images.

    Bias matching is a data filtering technique designed to eliminate shortcut learning. Generative AI models output images with rigid, predictable metadata: exact canvas dimensions (e.g.- 1024x1024) and consistent JPEG Quality Factors (QF). In contrast, real photos come in thousands of random resolutions and compression levels. If a raw dataset is fed to a deep learning model, the model neural network may quickly learn via a shortcut rule.

    Bias matching therefore forces real and fake images to share the exact same metadata profile, thereby eliminating any predictive signal from image metadata.

    Generative models cannot easiy change their output compression during dataset collection. Also, they are very consistent in their dimensions and encoding. Therefore we apply an asymetric bias matching approach:
      1. AI-generated images are left untouched as they already occupy a narrow, constrained band of space
      2. Any human-made images that fall outside the (`size` and `jpeg_qf` range) occupied by the fakes are removed
      3. A simple classifier is trained on metadata alone. If we get an accuracy close to 50% (akin to a random guess) shortcut learning is successfully eliminated. The higher the accuracy score, the more bias is inherent in the metadata (see `shortcut_probe()`)

    ****Consequence****: It is possible to remain with far less human-made images.

    Params:
      df (pd.DataFrame): The metadata/image dataset. Ensure columns `generator`, `height`, `width`, and `jpeg_qf` are present.
      cfg (class SubsetConfig): The bias matching configuration parameters. See `class SubsetConfig` for the structure. Can take in a `yaml` file in this structure.

    Returns:
     matched_df (pd.DataFrame): A dataframe with real and AI-generated images that are consistent across dimensions and JPEG quality factor (JPEG QF).
    """

    # Identifying real images
    is_real= df["generator"] == cfg.real_generator_token # Returns Boolean
    keep= pd.Series(True, index= df.index)

    # Minimum size (to be retained)
    if cfg.min_side is not None:
        keep &= ~is_real | ((df["width"] >= cfg.min_side) & (df["height"] >= cfg.min_side))

    # Maximum size
    if cfg.max_side is not None:
        keep &= ~is_real | ((df["width"] <= cfg.max_side) & (df["height"] <= cfg.max_side))

    # JPEG QF
    if cfg.jpeg_qf is not None and "jpeg_qf" in df.columns:
        lo = cfg.jpeg_qf - cfg.jpeg_qf_tolerance
        hi = cfg.jpeg_qf + cfg.jpeg_qf_tolerance
        keep &= ~is_real | df["jpeg_qf"].between(lo, hi)
          # Keeps images with JPEG QFs that are within a range
 
    return df[keep].copy() # Returns the filtered dataset with real images restricted

def shortcut_probe(df: pd.DataFrame, 
                   feature_cols: tuple[str, ...] = ("width", "height", "jpeg_qf"), 
                   n_folds: int = 5,
                   seed: int = 0) -> float:
    """
    This function checks whether real and AI-generated images can be distinguished without looking at the image itself. 

    A simple decision tree is fit on metadata only and returns 5-fold CV accuracy. 
      - ~0.50  -> metadata carries no label information. This is the score we want to achieve.
      - ~0.95  -> a model can 'solve' your benchmark without vision at all. Shortcut learning

    Params:
      df (pd.DataFrame): The image metadata dataset
      feature_cols (tuple[str, ...]): The input features to the simple classifier. Default features are `("width", "height", "jpeg_qf")` therefore ensure these are present in your `df` DataFrame otherwise specify the features explicitly.
      n_folds (int): Number C-V folds created. **Default**: 5
      seed (int): For reproducibility

    Returns:
      score (float): Average classification accuracy score across `n_folds`
    """

    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    cols= [c for c in feature_cols if c in df.columns]
    X= df[cols].fillna(-1).to_numpy() # Missing metadata is replaced with -1
    y= (df['generator'] != "nature").astype(int).to_numpy()
      # Real images: 0
      # AI generated (anything else): 1
    
    if len(np.unique(y)) < 2:
        return float("nan")
      # In the event the DataFrame contains only one type of image (either only real or AI-generated), "nan" is returned
    clf= DecisionTreeClassifier(max_depth= 3, random_state= seed)
    return float(cross_val_score(clf, X, y, cv= n_folds, n_jobs= -1).mean())
      # Returns average accuracy score across 5 corss-validation folds
      # We use a simple classifier as the goal is to identify where an obvious metadata shortcut is present

## ==================================================================================
## Image Data Splitting
## ==================================================================================
def assign_genimage_splits(df: pd.DataFrame, 
                           cfg: SubsetConfig, 
                           group_col: str = "group_id") -> pd.DataFrame:
    """
    Assigns the following train/validation/test splits:
      - train: Model training
      - val: For evaluation to find the best performing model
      - test_in_dist: Consists of images produced by the same generators that produced the images in the `train` set
      - test_ood_genimage: Images produced by completely new generators that were not in the `train` set

    Also handles how human-made images are split. The metadata manifest database should contain clusters of similar human-made and AI-generated images groups e.g.:
    ```
    group 123
      real original
      generated version A from generator X
      generated version B from generator Y
      generated version C from generator Z
    ```

    Basically, this function implements a group-aware splitting methodology where rather than splitting row-wise, it splits according to the number groups in the dataset. This therefore prevents the following:
      - Similar images from appearing in both training and val/test sets (data leakeage)
      - Real image duplication when specifiying the generators in the training and testing sets 

    Split order occurs as follows:
    ```
    OOD
    ↓
    test
    ↓
    validation
    ↓
    remaining → train
    ```
    """

    rng= np.random.default_rng(cfg.seed) # Deterministic randomness

    # Start with everything unassigned. The function then progressively assigns rows
    df= df.copy()
    df['split']= "unassigned"

    # Identify generator categories
    is_real= df['generator'] == cfg.real_generator_token # Real images
    is_ood_gen= df['generator'].isin(cfg.ood_generators) # OOD Generators
    is_train_gen= df['generator'].isin(cfg.train_generators) 
      # All generators eligible for train/val/test in distribution set
    
    # Collect real groups
    ood_test_split_real= 0.25 # Percentage of image groups to reserve for out of distribution testing
    real_groups = df.loc[is_real, group_col].dropna().unique() # Take a list of real groups
    rng.shuffle(real_groups) # Randomly shuffle them
    n_ood_real = int(len(real_groups) * ood_test_split_real)   
      # reserve a specified percentage and number of groups for OOD testing (real images)
    ood_real_groups = set(real_groups[:n_ood_real].tolist()) 
      # Set containing group Ids for real images for OOD testing
    df.loc[is_ood_gen, "split"] = "test_ood_genimage"
    df.loc[is_real & df[group_col].isin(ood_real_groups), "split"] = "test_ood_genimage"

    pool_mask = (is_train_gen | (is_real & ~df[group_col].isin(ood_real_groups)))
    pool_groups = df.loc[pool_mask, group_col].dropna().unique()
    rng.shuffle(pool_groups)
 
    n = len(pool_groups)
    n_test = int(n * cfg.test_in_dist_fraction)
    n_val = int(n * cfg.val_fraction)
    test_g = set(pool_groups[:n_test].tolist())
    val_g = set(pool_groups[n_test:n_test + n_val].tolist())
 
    df.loc[pool_mask & df[group_col].isin(test_g), "split"] = "test_in_dist"
    df.loc[pool_mask & df[group_col].isin(val_g), "split"] = "val"
    df.loc[pool_mask & (df["split"] == "unassigned"), "split"] = "train"
    return df
 

def stratified_pilot(
    df: pd.DataFrame,
    n_per_stratum: int,
    strata: tuple[str, ...] = ("split", "label", "generator"),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Deterministically down-sample the manifest for pipeline development.

    For development speed, it creates a smaller dataset while preserving important categories.

    A stratum is a subgroup defined by some combination of variables i.e.: `("split", "label", "generator")` which means groups such as:
    ```
    (train, real, nature)
    (train, fake, stable_diffusion)
    (train, fake, dalle)

    (val, real, nature)
    (val, fake, stable_diffusion)

    (test_in_dist, real, nature)
    ...
    ```
    The pilot takes up to `n_per_stratum` examples from each.
 
    Stratifying on (split, label, generator) guarantees the pilot is a
    miniature of the full set rather than an accidental pile of one generator.
    """
    cols = [c for c in strata if c in df.columns]
    return (
        df.groupby(cols, group_keys=False, observed=True)
          .apply(lambda g: g.sample(min(len(g), n_per_stratum), random_state=seed))
          .reset_index(drop=True))