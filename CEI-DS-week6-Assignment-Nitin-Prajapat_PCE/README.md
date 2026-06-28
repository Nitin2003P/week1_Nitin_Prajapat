# Autoencoder for Image Denoising using MNIST

## Project Overview

This project implements deep learning-based autoencoders to remove noise from handwritten digit images from the MNIST dataset.

Three different autoencoder architectures are implemented and compared:

1. FFNN Autoencoder
2. Transpose CNN Autoencoder
3. Upsampled CNN Autoencoder

---

## Dataset

* Dataset: MNIST Handwritten Digits Dataset
* Training Images: 60,000
* Test Images: 10,000
* Image Size: 28×28 pixels
* Color Format: Grayscale

---

## Project Objectives

* Add artificial noise to MNIST images.
* Train autoencoders to remove the noise.
* Compare the performance of different architectures.
* Visualize denoising results.
* Identify the best-performing model.

---

## Models Implemented

### 1. FFNN Autoencoder

Fully connected encoder-decoder network.

### 2. Transpose CNN Autoencoder

Convolutional encoder with transposed convolution decoder.

### 3. Upsampled CNN Autoencoder

Nearest-neighbor upsampling followed by convolution layers.

---

## Evaluation Metrics

* Mean Squared Error (MSE)
* Peak Signal-to-Noise Ratio (PSNR)
* Structural Similarity Index Measure (SSIM)

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib
* Scikit-image
* Jupyter Notebook

---

## Repository Structure

```text
MNIST-Autoencoder-Denoising/
├── README.md
├── autoencoder_mnist.ipynb
├── requirements.txt
├── models/
├── assets/
├── reports/
└── results/
```

---

## Author

Nitin Prajapat

B.Tech Computer Science and Engineering
Poornima College of Engineering, Jaipur
