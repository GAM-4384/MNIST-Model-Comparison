# MNIST-Model-Comparison

**Handwritten Digit Recognition: CNN vs SVM vs MLP on MNIST with PyQt5 GUI**

---

## Overview

This project implements and compares three different approaches to handwritten digit recognition on the MNIST dataset. A convolutional neural network, a support vector machine, and a multi-layer perceptron are trained independently and evaluated side by side. Two PyQt5 desktop applications are provided: one for single-model SVM inference and one for live side-by-side comparison of SVM and MLP predictions.

---

## Three Approaches

### CNN — `code1.py`
A two-block convolutional network built with TensorFlow/Keras. Each block contains two Conv2D layers with batch normalisation, ReLU activation, max pooling, and dropout. The flattened features pass through a 512-unit dense layer before the 10-class softmax output. Trained directly on raw 28×28 pixel arrays.

```
Input [28×28×1]
  → Conv2D(32) × 2 + BN + ReLU + MaxPool + Dropout(0.25)
  → Conv2D(64) × 2 + BN + ReLU + MaxPool + Dropout(0.25)
  → Dense(512) + BN + ReLU + Dropout(0.5)
  → Dense(10, softmax)
```

Callbacks: ModelCheckpoint, EarlyStopping (patience=10), ReduceLROnPlateau.
Saved as `mnist_cnn_model.h5`.

### SVM — `svm.py` / `svmtest.py`
Images are read from JPG files, flattened to 784-dimensional vectors, normalised to [0, 1], then reduced to 256 dimensions via PCA (retaining ~95% variance). An RBF-kernel SVM with one-vs-rest decision strategy is fitted on the reduced training set.

```
JPG → flatten [784] → normalise → PCA [256] → SVM (RBF, OvR)
```

Saved as `svm.model` + `pca.model`.

### MLP — `neural_network.py` / `test_neural_network.py`
Uses the same PCA preprocessing pipeline as SVM. An `MLPClassifier` with two hidden layers (100 → 50 neurons), ReLU activation, and Adam optimiser is trained on the reduced features.

```
JPG → flatten [784] → normalise → PCA [256] → MLP (100→50, ReLU, Adam)
```

Saved as `neural_network.model` + `nn_pca.model`.

---

## Project Structure

```
MNIST-Model-Comparison/
│
├── code1.py                    # CNN training and evaluation (TensorFlow)
├── svm.py                      # SVM training + shared image utilities
├── svmtest.py                  # SVM test-set evaluation
├── neural_network.py           # MLP training
├── test_neural_network.py      # MLP test-set evaluation
├── convert_mnist.py            # Convert raw idx binary files → JPG images
│
├── user_interface.py           # PyQt5 GUI: single image → SVM prediction
├── gui_comparison.py           # PyQt5 GUI: SVM vs MLP side-by-side comparison
│
├── mnist.npz                   # MNIST dataset (numpy format, for CNN)
├── mnist_cnn_model.h5          # Trained CNN weights
├── svm.model                   # Trained SVM classifier
├── pca.model                   # PCA model for SVM pipeline
├── neural_network.model        # Trained MLP classifier
└── nn_pca.model                # PCA model for MLP pipeline
```

---

## Data Preparation

The CNN (`code1.py`) reads directly from `mnist.npz`. The SVM and MLP pipelines read from JPG image files organised by label in the filename (`label_index.jpg`).

To generate the JPG files from raw MNIST binary data:

```bash
python convert_mnist.py
```

This parses the idx3-ubyte image files and idx1-ubyte label files and writes one JPG per sample named `{label}_{index}.jpg`.

---

## Training

```bash
# Train CNN
python code1.py

# Train SVM (reads from img_train/jpg_images/)
python svm.py

# Train MLP (reuses SVM image utilities)
python neural_network.py
```

---

## Evaluation

```bash
# Evaluate SVM on test set
python svmtest.py

# Evaluate MLP on test set
python test_neural_network.py
```

---

## Desktop Applications

**Single-model SVM interface** — upload any JPG digit image and get an instant SVM prediction:

```bash
python user_interface.py
```

**Side-by-side comparison interface** — displays SVM and MLP predictions simultaneously for a selected image, with colour-coded correct/incorrect labels. The "Test All Samples" button runs both models over the entire test set and reports accuracy for each:

```bash
python gui_comparison.py
```

---

## Model Comparison Summary

| Model | Input | Preprocessing | Framework |
|---|---|---|---|
| CNN | 28×28 pixel array | Normalise to [0,1] | TensorFlow/Keras |
| SVM | JPG → 784-dim vector | Normalise + PCA(256) | scikit-learn |
| MLP | JPG → 784-dim vector | Normalise + PCA(256) | scikit-learn |

CNN achieves the highest accuracy due to its spatial feature extraction capability. SVM and MLP share the same PCA preprocessing pipeline and can be compared directly in `gui_comparison.py`.

---

## License

This repository is released for academic and non-commercial use only.

Copyright © 2024 Merlin. All rights reserved.

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.
