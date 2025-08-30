# Deepfake Image Classification

This project focuses on classifying **deepfake images** into 5 categories using *supervised machine learning*. Two approaches were implemented:

1. Convolutional Neural Networks (CNNs) with PyTorch

2. Support Vector Machines (SVMs) with scikit-learn

The models were trained and evaluated on a dataset of:

- 12,500 training images

- 1,500 validation images

- 6,500 test images

## Models

#### Convolutional Neural Network (CNN)

Implemented in PyTorch (initially tested with Keras)

- Architecture evolved from a simple 3-layer CNN to a deeper model with BatchNormalization, MaxPooling, and Dropout.

- Data augmentations applied: normalization, rotations, crops, zoom, brightness/contrast adjustments, flips, and color jitter.

Optimization strategies:

- Adam / AdamW optimizers

- Cosine Annealing learning rate scheduling

- Label smoothing for stability

Best performance:

- 94.92% validation accuracy and 93.93% competition score

#### Support Vector Machine (SVM)

Implemented with scikit-learn.

- Preprocessing included brightness/contrast adjustment, normalization, PCA for dimensionality reduction.

- Kernel: Radial Basis Function (RBF).

Best performance:

- 61% validation accuracy and 59.5% competition score

## Requirements

- Python 3.9+

- PyTorch

- scikit-learn

- NumPy

- Matplotlib

- (Optional but highly recommended) CUDA-enabled GPU for CNN training
