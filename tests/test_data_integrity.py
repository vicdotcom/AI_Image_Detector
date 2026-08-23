"""
For testing image processing and metadata recording modules (`integrity.py`, `manifest.py`).

pHash Hamming Distance reference:
    - 0 to 5 differing bits: Almost certainly the same image (or minor edits/compression).
    - 6 to 10 differing bits: Similar content with moderate edits.
    - greater than 12 differing bits: Completely different images

Run with: pytest -q (See Pyproject for configuration)
"""

from __future__ import annotations

import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")) 
  # So that Python finds the modules located under /src directory

## Importing from created modules
from ai_detector.data.integrity import(sha256_file, probe_decodability, phash, hamming, group_ids_from_pairs, near_duplicate_pairs, estimate_jpeg_quality)

from ai_detector.data.manifest import(assert_no_leakage, LeakageError)


## Pytest fixtures
@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0) # Fixed seed


@pytest.fixture
def sample_image(rng) -> Image.Image: # Testing with a synthetic image
    """
    This creates a smooth, structured synthetic image. Random noise is a bad fixture for pHash as its pixels change rapidly due to random nature. Rapidly changing pixels mean high frequencies during DCT conversion. pHash relies on low-frequency information for encoding making the synthetic image a robust test for pHash.
    """
    y, x = np.mgrid[0:256, 0:256]
    arr = (128 + 100 * np.sin(x / 20) * np.cos(y / 30)).astype(np.uint8)
    return Image.fromarray(np.stack([arr,  # R
                                     arr.T, # G
                                     arr],  # B
                                     axis=-1))


## SHA-256 test --------------------------------------------------------------
def test_sha256_is_stable_and_content_sensitive(tmp_path):
    """
    Tests two properties of SHA-256
        - Same content -> same hash
        - Different content -> different hash
    """
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"hello")
    assert sha256_file(a) == sha256_file(b)
    b.write_bytes(b"hellp")
    assert sha256_file(a) != sha256_file(b)

## pHash test --------------------------------------------------------------
def test_phash_survives_resize_and_recompression(tmp_path, sample_image):
    """
    We check whether if two files representing approximately the same image will be similar perceptually. Small transformations of the same underlying image should not drastically change the perceptual hash.
    """
    original = tmp_path / "o.png"
    sample_image.save(original)
 
    resized = tmp_path / "r.jpg"
    sample_image.resize((200, 200)).save(resized, quality=70)
 
    d = hamming(phash(original), phash(resized))
    assert d <= 5, f"resize+recompress changed the hash by {d} bits"


def test_phash_separates_different_images(tmp_path, sample_image, rng):
    """
    Checks whether pHash can distinguish a genuinely different image
    """
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    sample_image.save(a)
    Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(b) 
      # Random noise image for comparison
    assert hamming(phash(a), phash(b)) > 10

## Corrupt Image Detection ----------------------------------------------------------
def test_corrupt_file_is_detected_not_raised(tmp_path, sample_image):
    """
    Test for `probe_decodability()`. A decodable image must have existing headers and metadata (`.verify()`) with readable pixels (`.load()`)
    """
    good = tmp_path / "good.jpg" # Decodable image
    sample_image.save(good, quality=90)
    data = good.read_bytes()
 
    bad = tmp_path / "bad.jpg" # Deliberately corrupt the image
    bad.write_bytes(data[: int(len(data) * 0.6)])
 
    assert probe_decodability(good).ok is True
    result = probe_decodability(bad)
    assert result.ok is False and result.error

## JPEG QF -----------------------------------------------------------
def test_jpeg_quality_estimate_tracks_encoder_setting(tmp_path, sample_image):
    """
    Infer the JPEG QF of an image. Our function should be accurate by +- 6 units
    """
    for q in (50, 75, 95):
        p = tmp_path / f"q{q}.jpg"
        sample_image.save(p, quality=q) # Creates three JPEGS with pre-stated QF
        est = estimate_jpeg_quality(p) # Estimates their quality
        assert est is not None
        assert abs(est - q) <= 6, f"estimated {est} for encoder quality {q}. Diff: {q - est}"


def test_png_has_no_jpeg_quality(tmp_path, sample_image):
    """
    Estimating JPEG QF should only be done on JPEG images
    """
    p = tmp_path / "x.png"
    sample_image.save(p)
    assert estimate_jpeg_quality(p) is None


## Transitive Grouping -------------------------------------------------------
def test_near_duplicate_grouping_is_transitive():
    """
    Testing dataset splitting logic. 

    If A~B and B~C then A, B and C must be in one group, or a cluster can still
    straddle the train/test boundary. Avoids train/test data leakage.
    """
    pairs = [(0, 1, 2), (1, 2, 3)]
    groups = group_ids_from_pairs(4, pairs)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]

## Banded LSH ----------------------------------------------------------
def test_banded_search_finds_close_pairs():
    """
    Where Banded LSH can actually find similar hashes within a dataset.

    See docstring for `near_duplicate_pairs()`
    """
    base = 0x0F0F0F0F0F0F0F0F
    hashes = [f"{base:016x}", 
              f"{base ^ 0b11:016x}", 
              f"{~base & (2**64 - 1):016x}"]
    pairs = near_duplicate_pairs(hashes, max_distance=3)
    found = {(a, b) for a, b, _ in pairs}
    assert (0, 1) in found
    assert (0, 2) not in found

## Laakage assertion ------------------------------------------------------
def test_leakage_assertion_fires():
    """
    Detects leakage signals. If an image with the same sha256 hash is in both the training and testing data.
    """
    df = pd.DataFrame({
        "sha256": ["aa", "aa", "bb"],
        "group_id": [1, 1, 2],
        "split": ["train", "test_in_dist", "train"],
    })
    with pytest.raises(LeakageError):
        assert_no_leakage(df) # We expect a LeakageError here


def test_leakage_assertion_passes_on_clean_split():
    df = pd.DataFrame({
        "sha256": ["aa", "bb", "cc"],
        "group_id": [1, 1, 2],
        "split": ["train", "train", "test_in_dist"],
    })
    assert_no_leakage(df)  # must not raise LeakageError