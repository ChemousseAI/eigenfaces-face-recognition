# Face Recognition using PCA, SVM and KNN

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)](#)
[![PCA](https://img.shields.io/badge/Dimensionality-PCA-green)](#)

🧑‍💻 Machine Learning | 👤 Face Recognition | 📊 PCA

---

## Overview

This project implements a classical Face Recognition system based on:

- Principal Component Analysis (PCA)
- Eigenfaces
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The objective is to reduce the dimensionality of facial images while preserving the most discriminative features and then perform face classification using machine learning algorithms.

---

## Dataset

The experiments were conducted using the Olivetti Faces Dataset.

### Dataset Characteristics

- 40 individuals
- 10 images per person
- Total images: 400
- Image size: 64 × 64 pixels
- Grayscale facial images

Dataset source:

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html

---

## Methodology

### 1. Data Preparation

The face images are loaded and reshaped into feature vectors containing:

4096 = 64 × 64 pixels

for each image.

### 2. Principal Component Analysis (PCA)

PCA is applied to:

- Reduce dimensionality
- Remove redundancy
- Extract Eigenfaces

Number of principal components:

```python
n_components = 150
```



### 3. Classification


The project evaluates two machine learning classifiers after PCA-based dimensionality reduction:

```python
# Support Vector Machine (SVM)
svm = SVC(
    kernel='rbf',
    C=1000,
    gamma=0.001
)

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(
    n_neighbors=2
)
```
### 4. Results


| Model     | Accuracy |
| --------- | -------- |
| PCA + SVM | 98%      |
| PCA + KNN | 93%      |

The PCA-SVM pipeline achieved the best recognition performance.


### 5. Future Improvements

- Hyperparameter optimization
- Deep learning approaches (CNNs)
- Real-time webcam face recognition
- Comparison with modern embeddings


