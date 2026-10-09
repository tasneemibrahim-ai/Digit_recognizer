# ✍️ Handwritten Digit Recognizer

A complete handwritten digit classification project using the **MNIST** dataset. This notebook implements **two distinct approaches** — a neural network built **from scratch with NumPy** and a deep learning model using **TensorFlow/Keras** — to classify 28×28 grayscale images of digits (0–9).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach 1 — Neural Network from Scratch (NumPy)](#approach-1--neural-network-from-scratch-numpy)
- [Approach 2 — Deep Neural Network (Keras)](#approach-2--deep-neural-network-keras)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)

---

## Overview

This project tackles the classic **Digit Recognizer** challenge. The goal is to correctly identify handwritten digits from pixel data. Two approaches are explored side-by-side in a single notebook to compare a low-level, from-scratch implementation with a high-level framework solution.

---

## Dataset

The project uses the **Kaggle MNIST** dataset:

| Split    | Samples | Features                   |
|----------|---------|----------------------------|
| Training | 42,000  | 785 (1 label + 784 pixels) |
| Testing  | 28,000  | 784 (pixels only)          |

Each image is a **28 × 28** grayscale pixel grid, flattened into a 784-dimensional vector. Pixel values range from 0 to 255 and are **normalized to [0, 1]** before training.

> **Source:** [Kaggle — Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)

---

## Approach 1 — Neural Network from Scratch (NumPy)

A fully-connected, 2-layer neural network implemented **entirely with NumPy** — no deep learning framework involved.

### Architecture

```
Input (784) → Dense (10, ReLU) → Dense (10, Softmax) → Output
```

### Key Implementation Details

| Component                | Detail                                                                                      |
|--------------------------|---------------------------------------------------------------------------------------------|
| **Weight Initialization**| Xavier initialization for stable gradient flow                                               |
| **Forward Propagation**  | Z₁ = W₁·X + b₁ → A₁ = ReLU(Z₁) → Z₂ = W₂·A₁ + b₂ → A₂ = Softmax(Z₂)                    |
| **Loss Function**        | Cross-entropy with **L2 regularization** (λ = 0.01)                                         |
| **Backward Propagation** | Manual gradient computation with regularization-aware weight updates                         |
| **Optimizer**            | Vanilla gradient descent (α = 0.10, 500 iterations)                                         |

### Training

- Data is split into **41,000 training** and **1,000 dev** samples.
- Loss and accuracy are logged every 10 iterations, with training curves plotted for both.

---

## Approach 2 — Deep Neural Network (Keras)

A deeper network using TensorFlow/Keras with **dropout regularization**.

### Architecture

```
Input (784)
  → Dense (1500, ReLU)
  → Dropout (0.5)
  → Dense (1000, ReLU)
  → Dropout (0.5)
  → Dense (500, ReLU)
  → Dropout (0.5)
  → Dense (10, Softmax)
  → Output
```

### Key Implementation Details

| Component       | Detail                                    |
|-----------------|-------------------------------------------|
| **Optimizer**   | Adam                                      |
| **Loss**        | Sparse Categorical Cross-Entropy          |
| **Regularization** | Dropout (50%) after each hidden layer |
| **Train/Test Split** | 90% / 10% (via `train_test_split`)  |

### Training Visualizations

- Loss vs. epochs (train & validation)
- Accuracy vs. epochs (train & validation)

The trained Keras model is saved as **`handwritten.h5`** for later inference.

---

## Results

| Model               | Validation / Dev Accuracy |
|----------------------|---------------------------|
| NumPy from Scratch   | ~85%                      |
| Keras Deep Network   | ~97%+                     |

> *Exact accuracy varies with random seed and number of epochs.*

---

## Project Structure

```
Digit_recognizer/
├── Digit_recognizer.ipynb   # Main notebook (both approaches)
├── README.md                # This file
├── train.csv                # Training data (from Kaggle)
├── test.csv                 # Test data (from Kaggle)
└── handwritten.h5           # Saved Keras model (generated after training)
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Digit_recognizer.git
cd Digit_recognizer
```

### 2. Download the dataset

Download `train.csv` and `test.csv` from [Kaggle — Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data) and place them in the project root.

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### 4. Run the notebook

```bash
jupyter notebook Digit_recognizer.ipynb
```

---

## Dependencies

| Package        | Purpose                         |
|----------------|----------------------------------|
| `numpy`        | Matrix operations & from-scratch NN |
| `pandas`       | Data loading & manipulation      |
| `matplotlib`   | Training curve & sample visualization |
| `seaborn`      | Enhanced data visualizations     |
| `scikit-learn` | Train/test splitting             |
| `tensorflow`   | Keras deep learning model        |
