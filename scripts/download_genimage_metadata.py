#!/usr/bin/env python3
"""
Downloads the GenImage metadata CSV (and corrupted file list) from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FAKDIHF). See [API Documentation](https://guides.dataverse.org/en/latest/api/dataaccess.html) for reference.

Usage: 
    python scripts/download_genimage_metadata.py --dest data/raw/genimage_meta

On downloading the data, the script records the following into a JSON file (`provenance.json`):
    - What was downloaded
    - When it was downloaded
    - Where it came from
    - How many bytes it contains
    - What SHA-256 hash it has
    - What checksum Dataverse reports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

DATAVERSE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/AKDIHF"
  # No API key needed as the data is publicly accessible
  # The DOI is to be passed to the Dataverse API to identify the dataset programmatically.

HEADERS = {
    "User-Agent": "AI-Image-Detector-Research/0.1 (educational project; contact: vradeny@gmail.com)"
          }  # For user verification

## Required files
WANTED = ("metadata.csv", "corrupted_files.txt", "class_map")

def main() -> int:
    ap= argparse.ArgumentParser(description= __doc__)
    ap.add_argument("--dest",  type= Path, default= "data/raw/genimage_meta")
    ap.add_argument("--doi", default= DOI)
    ap.add_argument("--all", action= "store_true", 
                    help= "Downloads every file, including the entire ~500GB GenImage image data. May not be the most ideal option")
     # "store_true" makes this argument optional
    
    args= ap.parse_args()

    args.dest.mkdir(parents= True, exist_ok= True) # Creates the destination directory
      # parents=True means it can create missing parent directories too.
      # exist_ok=True don't complain if the directory already exists.

    files = list_dataset_files(args.doi)
    print(f"Dataset {args.doi} exposes {len(files)} files.")
    
    provenance = {
        "doi": args.doi, # Dataset identifier
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": DATAVERSE, # Source URL
        "files": [],
    }

    for entry in files:
        label = entry.get("label", "") # File label (name)
        data_file = entry.get("dataFile", {}) # File ID
        if not args.all and not any(w in label.lower() for w in WANTED):
            continue # Skip unwanted files

        size = data_file.get("filesize")
        print(f"- {label} ({(size or 0) / 1e6:.1f} MB)")
        out = download_file(data_file["id"], args.dest / label, size) # Downloads the file
        digest = sha256_file(out) # Computes SHA-256 hash for the downloaded file

        # Record info in provenance dictionary
        provenance["files"].append({
            "label": label,
            "bytes": out.stat().st_size,
            "sha256": digest,
            "server_checksum": data_file.get("checksum"),
        })
        print(f"  sha256 = {digest}")

    if not provenance["files"]:
        print("No matching files found. Run with --all to inspect the listing, "
            "or check the WANTED patterns at the top of this script.",
            file=sys.stderr)
        return 1

    prov_path = args.dest / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"\nWrote provenance record -> {prov_path}")
    print("COMMIT provenance.json. Do not commit the CSV itself.")
    return 0

## Hashing downloaded files 
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """
    Assigns a SHA-256 hash fingerprint to a downloaded file so that we can know if the file changes in any way when downloading the same file at another time.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block) # Process the file step by step (per 1<<20 = 1MB) rather than all at once (streaming)
    return h.hexdigest()


def list_dataset_files(doi: str) -> list[dict]:
    """
    Ask Dataverse for the file listing of a dataset version.
 
    Returns a list of dicts with at least 'label' (filename) and 'dataFile'
    (which contains the numeric id used for download, plus the server's own
    md5/checksum that we can verify against).
    """
    url = f"{DATAVERSE}/api/datasets/:persistentId/versions/:latest" # API url construction
    resp = requests.get(url, params={"persistentId": doi}, timeout=60, headers= HEADERS)
    resp.raise_for_status()
    return resp.json()["data"]["files"]


def download_file(file_id: int, dest: Path, expected_size: int | None = None) -> Path:
    """
    Progressively stream a datafile to disk, skipping if a complete copy already exists.

    :params:
        file_id (int): Tells Dataverse which file to return
        dest (Path): Destination path where the file should be downloaded
        expected_size (int | None): File checker for the function where if the file is already downloaded, it can skip re-downloading it if the file sizes match. 
    """
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
         # If the downloaded file exists and its size is the same as the file in Dataverse
        print(f"  [skip] {dest.name} already present and correct size")
        return dest
            # This makes the script idempotent such that running the same command again should not unnecessarily repeat the work
 
    url = f"{DATAVERSE}/api/access/datafile/{file_id}" # Download API URL
    with requests.get(url, stream=True, timeout=300, headers= HEADERS) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part") # Adds ".part" for incomplete files
        written = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20): # File streaming per 1 MB
                f.write(chunk)
                written += len(chunk)
                print(f"\r  {dest.name}: {written / 1e6:7.1f} MB", end="", flush=True)
        print()
        tmp.replace(dest)   # Fully downloaded files are renames appropriately
    return dest


if __name__ == "__main__":
    main()