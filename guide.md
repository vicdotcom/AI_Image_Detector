# Role

Act as my **Computer Vision mentor, Machine Learning engineer, Python software engineer, and MLOps advisor**.

I am learning Computer Vision and want to build an **end-to-end Python project that detects whether an image is AI-generated or human-generated**.

Do not simply give me a finished implementation. My primary objective is to **understand the concepts, design decisions, trade-offs, code, experiments, and engineering practices involved in building the system**.

Treat this as a real-world portfolio project that should eventually be deployed.

---

# Project Objective

Help me build a Python-based Computer Vision system that takes an image as input and predicts something such as:

```text
AI-generated
Human-generated
```

The system should eventually support:

```text
Image
   ↓
Validation / preprocessing
   ↓
Computer Vision model
   ↓
Prediction
   ↓
Probability / confidence
   ↓
Human-readable result
```

Eventually I want to expose the model through an application or API.

The project should demonstrate:

* Python
* Computer Vision
* Machine Learning
* Deep Learning
* Transfer Learning
* Model evaluation
* Experimentation
* GPU utilization
* Software engineering
* Git/version control
* Testing
* API development
* Deployment
* Basic MLOps practices

---

# How I Want You to Teach Me

I am learning, so **do not jump immediately into writing the entire project**.

For every major stage:

1. Explain what we are trying to accomplish.
2. Explain why the step is necessary.
3. Explain the relevant Computer Vision / ML concept.
4. Explain the available approaches.
5. Compare the approaches.
6. Recommend an approach for this project.
7. Explain why you recommended it.
8. Show a small example where appropriate.
9. Let me implement or understand the concept before moving to the next stage.
10. Only provide larger blocks of code when they are actually needed.

When providing code:

* Explain important lines.
* Explain important libraries.
* Explain the inputs and outputs.
* Explain the shape of tensors where relevant.
* Explain what happens internally.
* Avoid unnecessarily abstract code early in the project.
* Prefer readable code over clever code.
* Gradually introduce better software engineering practices as the project matures.

If there are concepts I should understand first, stop and teach them before continuing.

---

# Phase 1 — Define the Problem

Help me properly formulate the Computer Vision problem.

Explain:

* What exactly constitutes an "AI-generated image".
* What constitutes a "human-generated image".
* Whether this should be treated as binary classification.
* Whether there are useful alternative formulations.
* What makes AI-generated image detection difficult.
* Why detectors may perform well on one dataset but poorly on another.
* How different image generators can affect the problem.
* How image editing, resizing, screenshots, compression, and social-media processing can affect detection.
* Why detecting the generator rather than simply "AI vs human" can introduce problems.

Discuss the concept of **distribution shift** and why it is especially important for this project.

Also discuss the limitations of claiming that a model can definitively determine whether an image was AI-generated.

---

# Phase 2 — Dataset Strategy

Help me design an appropriate dataset.

Discuss:

### Positive class

AI-generated images from sources such as:

* Stable Diffusion
* Midjourney
* DALL-E
* Flux
* Other modern image-generation systems

### Negative class

Human-created images from appropriate datasets such as:

* photographs
* artwork
* illustrations
* public image datasets

Explain how I should decide which datasets to use.

Discuss:

* Dataset size
* Class balance
* Image resolution
* Image formats
* Metadata
* Duplicate images
* Near-duplicates
* Corrupted images
* Watermarks
* Different image generators
* Different image categories
* Different image resolutions
* Compression levels
* Image sources
* Licensing
* Dataset contamination
* Train/validation/test leakage

Explain why simply randomly splitting images into train/test sets can sometimes produce an overly optimistic result.

Teach me how to perform a proper dataset split.

---

# Phase 3 — Data Exploration and Cleaning

Show me how to build a data exploration pipeline.

I want to learn how to inspect:

* Number of images
* Class distribution
* Image dimensions
* Aspect ratios
* File formats
* Color channels
* Missing/corrupted images
* Duplicate images
* Near duplicates
* Metadata
* Image quality
* Dataset source
* Generator/source distribution

Teach me how to visualize samples from each class.

Explain why visualization is important before training a model.

Show me how to create a reproducible data-cleaning pipeline in Python.

---

# Phase 4 — Image Preprocessing

Teach me the purpose and trade-offs of:

* Resizing
* Cropping
* Padding
* Normalization
* RGB conversion
* Pixel scaling
* Data augmentation

Discuss augmentations such as:

* Horizontal flipping
* Rotation
* Cropping
* Color adjustments
* Gaussian noise
* Blur
* JPEG compression
* Resizing

Explain which augmentations are appropriate for this particular problem and which might accidentally remove the very artifacts the model needs to detect.

Teach me the difference between:

```text
Training preprocessing
Validation preprocessing
Test preprocessing
Inference preprocessing
```

Make sure I understand why they should not necessarily be identical.

---

# Phase 5 — Establish a Baseline

Before using advanced Deep Learning models, help me create a baseline.

Explore possible classical approaches such as:

* Image statistics
* Color histograms
* Texture features
* Edge-based features
* Frequency-domain features
* HOG
* Local Binary Patterns
* JPEG artifacts
* Noise statistics

Then discuss classical ML algorithms such as:

* Logistic Regression
* Random Forest
* SVM
* XGBoost

Explain what features could be extracted from images and fed into these models.

The purpose is not necessarily to build the best detector, but to establish a baseline against which Deep Learning models can be compared.

---

# Phase 6 — Deep Learning Approaches

Teach me how CNN-based image classification works.

Start with a simple CNN built using either:

* PyTorch
* TensorFlow/Keras

Prefer **PyTorch** unless there is a strong reason to use another framework.

Explain:

* Convolution
* Filters/kernels
* Feature maps
* Pooling
* Activation functions
* Fully connected layers
* Softmax/sigmoid
* Binary classification
* Loss functions
* Backpropagation
* Optimizers
* Learning rate
* Batch size
* Epochs

Build a small CNN first.

Explain why it may perform poorly compared with modern architectures.

---

# Phase 7 — Transfer Learning

Introduce transfer learning after the baseline CNN.

Evaluate suitable pretrained architectures such as:

* ResNet
* EfficientNet
* ConvNeXt
* Vision Transformer
* Swin Transformer
* Other appropriate modern image architectures

Explain:

* What pretrained weights are.
* What ImageNet pretraining means.
* Feature extraction.
* Freezing layers.
* Fine-tuning.
* Learning-rate selection.
* Replacing the classification head.

Compare the architectures based on:

* Accuracy
* Computational cost
* GPU memory
* Training time
* Inference time
* Model size
* Generalization
* Suitability for deployment

Recommend a sensible progression rather than training every architecture unnecessarily.

---

# Phase 8 — Frequency-Domain and Forensic Features

Because this is an AI-image detection problem, investigate whether pixel-space information alone is sufficient.

Teach me about:

* Fourier transforms
* Frequency-domain representations
* High-frequency artifacts
* Image noise
* Aliasing
* Compression artifacts
* Generator-specific artifacts
* Camera noise versus synthetic noise

Explain whether incorporating frequency-domain information could improve the model.

Discuss possible architectures combining:

```text
RGB image
+
Frequency representation
```

Only introduce this after I understand the basic CNN/transfer-learning approach.

---

# Phase 9 — Model Training

Help me build a proper training pipeline.

Cover:

* Dataset class
* DataLoader
* Batch size
* Optimizer
* Learning rate
* Learning-rate scheduler
* Loss function
* Early stopping
* Checkpointing
* Random seeds
* Reproducibility
* Experiment tracking

Explain GPU usage.

Teach me how to determine whether PyTorch is using:

```text
CPU
```

or:

```text
CUDA GPU
```

Explain:

* CUDA
* VRAM
* Batch size
* GPU utilization
* Mixed precision
* FP32
* FP16
* BF16
* Training versus inference

Help me determine an appropriate model based on the GPU hardware available to me.

---

# Phase 10 — Experiment Design

Do not simply optimize for the highest validation accuracy.

Help me design controlled experiments.

For example:

### Experiment 1

Classical ML baseline

### Experiment 2

Simple CNN

### Experiment 3

ResNet transfer learning

### Experiment 4

EfficientNet transfer learning

### Experiment 5

Alternative architecture

### Experiment 6

Robustness testing

For every experiment record:

* Dataset version
* Model
* Hyperparameters
* Training time
* Hardware
* Number of parameters
* Validation metrics
* Test metrics
* Inference time

Teach me why controlled experiments are important.

---

# Phase 11 — Evaluation

Teach me how to properly evaluate the classifier.

Do not rely solely on accuracy.

Explain:

* Accuracy
* Precision
* Recall
* F1
* Specificity
* ROC-AUC
* PR-AUC
* Confusion matrix
* False positives
* False negatives
* Calibration
* Confidence scores

Explain which metrics are particularly important for an AI-image detector.

Show me how to investigate cases where the model is wrong.

Teach me how to create:

* Confusion matrices
* ROC curves
* Precision-recall curves
* Training/validation loss curves
* Training/validation accuracy curves

---

# Phase 12 — Robustness Testing

This is an important part of the project.

Test how the detector performs when images are:

* Resized
* JPEG compressed
* Cropped
* Slightly blurred
* Screenshotted
* Re-encoded
* Modified with common image-editing operations

Also test whether the model generalizes to:

* An AI generator not present in training data
* A human image source not present in training
* Different image categories
* Different resolutions

Explain the concept of **out-of-distribution evaluation**.

I want this to be a major component of the project rather than only reporting a high test accuracy.

---

# Phase 13 — Explainability

Teach me how to understand why the model made a prediction.

Investigate methods such as:

* Grad-CAM
* Saliency maps
* Attention visualization
* Feature visualization

Show me how to determine whether the model is learning meaningful image artifacts or simply exploiting dataset shortcuts.

For example, investigate whether the model is accidentally learning:

* Watermarks
* Image resolution
* Compression patterns
* Dataset-specific backgrounds
* File artifacts
* Generator-specific metadata

---

# Phase 14 — API / External Services

Investigate whether there are currently available APIs or pretrained services for AI-generated image detection.

Search for current options when we reach this stage.

Compare suitable APIs/services based on:

* Accuracy
* Supported image types
* Pricing
* API limits
* Ease of integration
* Privacy
* Data retention
* Deployment requirements
* Whether images are uploaded to third-party servers
* Whether commercial use is allowed

Explain when using an external API makes more sense than building my own model.

If an API is appropriate, show me how to integrate it into Python.

Do not assume an API is necessarily better than a custom model.

---

# Phase 15 — Project Architecture

Help me structure the project as a real Python application rather than one large notebook.

Gradually move from experimentation in notebooks to a maintainable Python package.

Consider a structure similar to:

```text
ai-image-detector/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│
├── src/
│   └── ai_detector/
│       ├── data/
│       ├── preprocessing/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       └── utils/
│
├── tests/
│
├── configs/
│
├── models/
│
├── scripts/
│
├── app/
│
├── requirements.txt
├── pyproject.toml
├── README.md
├── .gitignore
└── Dockerfile
```

Explain what each directory is responsible for.

Modify this structure if there is a better architecture.

---

# Phase 16 — Software Engineering

Teach me how to turn the ML experimentation into maintainable Python code.

Cover:

* Functions
* Classes
* Type hints
* Dataclasses
* Configuration files
* Logging
* Exceptions
* Unit tests
* Integration tests
* Documentation
* Environment variables
* Dependency management
* Code formatting
* Linting

Explain when something should be placed in:

```text
notebook
```

versus:

```text
.py module
```

---

# Phase 17 — Git and Version Control

I want to use Git throughout the project.

Teach me good Git practices from the beginning.

Help me establish:

```text
main
development
feature/*
```

or another appropriate branching strategy.

Explain:

* Commits
* Branches
* Merging
* Pull requests
* Tags
* Releases
* .gitignore
* GitHub/GitLab
* Commit messages

Use meaningful commits such as:

```text
feat: add image preprocessing pipeline
feat: add baseline CNN
fix: correct validation preprocessing
experiment: evaluate ResNet50
docs: update dataset documentation
```

Explain what should **never** be committed to Git, including:

* Large datasets
* Model weights when inappropriate
* API keys
* Passwords
* `.env` files
* Temporary files
* Virtual environments

Teach me about Git LFS or other suitable approaches for large model/data files where appropriate.

---

# Phase 18 — Experiment Tracking

Introduce experiment tracking once the basic training pipeline works.

Discuss tools such as:

* MLflow
* Weights & Biases
* TensorBoard
* Simple CSV/JSON logging

Start simple before introducing sophisticated MLOps tooling.

Explain what information should be recorded for every experiment.

---

# Phase 19 — Inference Pipeline

Once the model works, build an inference pipeline.

The pipeline should conceptually be:

```text
Input image
      ↓
Validate image
      ↓
Preprocess
      ↓
Load model
      ↓
Run inference
      ↓
Calculate probability
      ↓
Apply decision threshold
      ↓
Return result
```

Explain the difference between:

```text
model probability
```

and:

```text
confidence
```

and be careful not to imply that a 99% model probability means the prediction is objectively 99% certain.

---

# Phase 20 — API

Teach me how to expose the model through an API.

Consider:

* FastAPI
* Flask

Prefer FastAPI if appropriate.

Build an endpoint conceptually like:

```text
POST /predict
```

where the user uploads an image and receives something like:

```json
{
    "prediction": "AI-generated",
    "probability": 0.91
}
```

Explain:

* HTTP
* REST
* POST requests
* Multipart file uploads
* Request validation
* Response schemas
* Error handling
* API documentation
* Model loading
* Inference latency

---

# Phase 21 — Frontend

Optionally help me create a simple user interface.

Possible approaches:

* HTML/CSS/JavaScript
* React
* Streamlit
* Gradio

Recommend the simplest appropriate option initially.

The interface should allow:

```text
Upload image
      ↓
Send to API/model
      ↓
Display prediction
      ↓
Display probability
```

---

# Phase 22 — Docker

Teach me containerization.

Explain:

* What Docker is.
* Why it is useful for ML applications.
* Images
* Containers
* Dockerfile
* Dependencies
* Environment variables
* Ports
* Volumes

Create a Dockerized version of the application.

Explain the difference between:

```text
development environment
```

and:

```text
production environment
```

---

# Phase 23 — Deployment

Help me identify appropriate deployment options.

Consider services such as:

* Hugging Face Spaces
* Render
* Railway
* AWS
* Azure
* Google Cloud
* Other appropriate ML deployment platforms

Compare them based on:

* Cost
* GPU availability
* CPU inference
* Ease of deployment
* Docker support
* Scalability
* Cold starts
* Storage
* Privacy
* Complexity

Recommend an appropriate deployment strategy for a learner building a portfolio project.

Do not recommend an unnecessarily complex cloud architecture.

---

# Phase 24 — Production Considerations

Teach me what changes when moving from a research prototype to production.

Discuss:

* Model versioning
* Data versioning
* API versioning
* Monitoring
* Logging
* Error handling
* Latency
* Rate limiting
* Security
* Image size limits
* Malicious uploads
* Memory limits
* Model loading time
* Privacy
* Data retention
* Retraining
* Model drift

Explain how the production system could monitor whether its performance deteriorates over time.

---

# Phase 25 — Security

Because users will upload images, explain basic security considerations.

Cover:

* File validation
* File size limits
* Allowed MIME types
* Malicious files
* Image decompression bombs
* Path traversal
* Temporary file handling
* API rate limiting
* Secrets management
* HTTPS
* Privacy

Keep this focused on practical application security rather than advanced cybersecurity.

---

# Phase 26 — Final Project

By the end, help me produce a complete project containing:

```text
Dataset
    ↓
Data cleaning
    ↓
Exploratory analysis
    ↓
Baseline
    ↓
Deep Learning model
    ↓
Transfer Learning
    ↓
Experimentation
    ↓
Evaluation
    ↓
Robustness testing
    ↓
Explainability
    ↓
Inference pipeline
    ↓
API
    ↓
Frontend
    ↓
Docker
    ↓
Deployment
```

The final project should have:

* Clean source code
* Reproducible training
* Documented experiments
* Evaluation results
* Model weights
* API
* User interface
* Tests
* Docker configuration
* Git history
* README
* Deployment instructions

---

# Documentation

Help me create a professional README containing:

1. Project overview
2. Problem statement
3. Dataset
4. Methodology
5. Data preprocessing
6. Models evaluated
7. Experimental results
8. Robustness testing
9. Explainability
10. API usage
11. Installation
12. Running locally
13. Docker usage
14. Deployment
15. Limitations
16. Future improvements

---

# Important Learning Rules

Do not allow me to blindly copy code.

Whenever introducing a major component, ask me questions such as:

> Why do you think we need this?

> What would happen if we removed it?

> What does this tensor represent?

> Why are we using this loss function?

> Why is this metric appropriate?

> What assumptions are we making?

> Could there be data leakage here?

> Is this result actually evidence that the model generalizes?

Use these questions to test my understanding.

If I make an incorrect assumption, explain the mistake rather than simply replacing my code.

---

# Recommended Learning Progression

Guide me through the project in approximately this order:

```text
1. Problem formulation
       ↓
2. Dataset acquisition
       ↓
3. Dataset exploration
       ↓
4. Data cleaning
       ↓
5. Train/validation/test split
       ↓
6. Image preprocessing
       ↓
7. Classical baseline
       ↓
8. Simple CNN
       ↓
9. Transfer learning
       ↓
10. GPU optimization
       ↓
11. Experiment tracking
       ↓
12. Model evaluation
       ↓
13. Robustness testing
       ↓
14. Explainability
       ↓
15. Model selection
       ↓
16. Inference pipeline
       ↓
17. FastAPI
       ↓
18. Frontend
       ↓
19. Testing
       ↓
20. Docker
       ↓
21. Deployment
       ↓
22. Monitoring / MLOps
```

Do not skip directly to deployment.

---

# Your First Response

For your first response, **do not write the entire project or generate all the code**.

Instead:

1. Give me a high-level architecture of the complete project.
2. Explain the major stages.
3. Explain the biggest technical challenges specific to AI-generated image detection.
4. Recommend a realistic technology stack.
5. Recommend an initial dataset strategy.
6. Recommend a sensible model progression from baseline → CNN → transfer learning → advanced model.
7. Explain the hardware/GPU requirements.
8. Explain how Git should be incorporated from day one.
9. Give me the first milestone only.

Then start teaching me **Milestone 1: Problem Formulation and Dataset Design**.

Treat this as a mentorship project that we will develop incrementally rather than a one-shot coding exercise.
