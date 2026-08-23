## Manifest Creation
### Introduction
We create a Manifest which contains various image statistics and metadata information to guide on data splitting, and mitigation of data leakage and/or shortcut learning by utilizing three main modules:
```bash
└── src/ai_detector
    └── data
        ├── integrity.py  # For computing image statistics
        ├── manifest.py   # Creates the manifest dataset file and stores image statistics
        └── selection.py  # Uses the manifest dataset to split image data into train/val/test sets
```

### Image Statistics (`integrity.py`)
Raw images are first processed by `integrity.py` which assess the following:
  - Whether the files are exactly the same (Cryptographic hashing)
  - Whether the image file can be opened and is readable (Image decoding)
  - Whether the images look similar (Perceptual hashing)
  - How the image was encoded (JPEG Quantization)
  - Which images belong to the same duplicate or near-duplicate groups (Union-Find)

#### Cryptographic Hashing (Duplicate Detection)
We use a `sha-256` cryptographic hash function which converts raw images into a fixed 265-bit (64-character) hexadecimal digital fingerprint. It is physically impossible for two images to have the same output hash which means any images with the same hash value are therefore exactly identical. Beneficial for this pipeline due to its compute speed as it does not need to parse or decode the image file.

#### Image Decoding
We check whether an image is fully decodable. A decodable image is simply an uncorrupted file that can be opened and read.

An image file (i.e.- .jpg, .png) is normally structured as follows.
  - Header and Metadata- Dimensions (1920x1080), color profiles, EXIF (technical image details such as location, time and date, camera settings, device info)
  - Compressed pixel payload

An image can therefore fail to decode via:
  - Header Corruption- The image file is garbage or the file may not even be an image at all
  - Truncation or Payload Corruption- The header is perfectly fine, but the pixel stream is broken or missing (i.e.- The full image is not loaded). Can occur due to network connections dropping halfway through a download or web scrape.

Therefore, a two-stage check is applied to ensure each aspect of the image is decodable:
  - First checking image headers and metadata
  - Decompressing each pixel per image to catch truncated downloads or corrupted byte streams before training crashes occur.

#### Perceptual Hashing
Perceptual hashing (`pHash`) creates a unique fingerprint of an image based on its visual similarity with other images; unlike SHA-256 hashing which only tracks exact image duplicates. The rationale for this operation is that say Image A is the photograph of a dog. Image B may be the same photograph though resized, cropped, made brighter/darker, and so on....

To serve as input into the `pHash` function we implement a Discrete Cosine Transform (DCT) on all images. This is a mathematical method that changes an image from the spatial domain (raw pixel values) to the frequency domain (cosine wave components) where pixels are placed in groups. It essentially splits the image into high and low frequency parts where:
  - Low frequencies (Top left)- Represent broad patterns, shaped, smooth gradients, overall illumination, large scale structure. This is where the bulk of the image's visual information is found.
  - High frequencies (Bottom right)- Represents edges, fine textures, sharp details, rapid color transitions. Since DCT works as an image compression tool, these high frequencies are what can be discarded to save storage space without noticeably distorting the image.
  

A DCT transformation is applied to each image and an 8x8 frequency matrix grid containing the lowest frequencies are selected as shown below then hashed to return a hexadecimal hash (16-chatacter hexadecimal string).
```ascii
    ┌────────────────────┐
    │ █ █ █ ░ ░ ░ ░ ░    │
    │ █ █ █ ░ ░ ░ ░ ░    │
    │ █ █ █ ░ ░ ░ ░ ░    │
    │ ░ ░ ░              │
    │ ░ ░ ░              │
    │                    │
    └────────────────────┘
```

The core operation that detects similar images is **Hamming Distance** computation between any two pHashes.

Hamming Distance is a metric that measures how different two equal-length sequences (strings (i.e.- hashes), bit arrays, vectors, etc) are from each other. From the pHashes obtained:
  - 0 to 5 differing bits: Almost certainly the same image (or minor edits/compression).
  - 6 to 10 differing bits: Similar content with moderate edits.
  - greater than 12 differing bits: Completely different images


#### JPEG Quantization
**JPEG Quality Factor (QF)** is a numerical value (typically scaled from 0 to 100) that determines the level of compression applied when saving a JPEG image. Higher values mean higher visual quality and larger file sizes, while lower values discard more visual data to save space. 

We estimate the JPEG QF that was applied when the image was initially encoded at its source. In practice, image data could come from various sources. If we scraped the AI-generated image set from one website and the "human" set from another website, then those two websites likely have two different re-encoding pipelines. The model may then try to "cheat" by finding the easiest separable signal in the training distribution (i.e.- the encoder signal). Estimating the JPEG QF allows us to mitigate this risk.

It also allows for robustness against real-world pipelines. Real-world images uploaded to various such as WhatsApp, Twitter, or Instagram undergo automatic re-compression and downsampling at low/medium quality factors. If we were to train the model only on pristine, high-QF images, it could fail when deployed against compressed images found on the web.

#### Grouping Similar Images (Banded LSH)
Similar images with narrow Hamming distance are grouped together. Hamming distance can only be computed between image paire. Therefore, rather than simply performing pairwise comparisons between images can result to millions of combinations, we implement Banded Locality-Sensitive Hashing (LSH) which is an algorithmic technique used to find approximate nearest neighbours or similar items in massive datasets.


### Conceptual Flow
The above image statistics are then recorded into a metadata dataset via `manifest.py`. `selection.py` draws from the created manifest to construct train/validation/test splits.

```text
Raw image collection
       │
       ▼
   integrity.py
       │
       ▼
Check files / hashes / grouping
       │
       ▼
   `manifest.py`
       │
       ▼
Create metadata manifest
       │
       ▼
  `selection.py`
       │
       ├── Normalize manifest column names
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