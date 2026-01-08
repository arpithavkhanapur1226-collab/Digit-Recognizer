# Digit-Recognizer
Handwritten Digit Recognition using CNN

Project Overview

This project focuses on building a Handwritten Digit Recognition System using a Convolutional Neural Network (CNN).
The model is trained on the MNIST digit dataset (via Kaggle’s Digit Recognizer competition) and is capable of predicting digits (0–9) from grayscale images.
The trained model is further deployed as a web application using Flask, allowing users to upload an image and receive a predicted digit in real time.
 Objectives
To understand image classification using CNNs
To train and validate a deep learning model on handwritten digit data
To generate predictions for unseen test data
To deploy the trained model as a simple web application

Technologies Used

Python
TensorFlow / Keras
NumPy, Pandas
Matplotlib, Seaborn
Scikit-learn
Flask (for deployment)
Kaggle (Digit Recognizer Competition)

Dataset
Source: Kaggle – Digit Recognizer (MNIST)
Training Data: 42,000 labeled images
Test Data: 28,000 unlabeled images
Image Size: 28 × 28 pixels (grayscale)
Each image represents a handwritten digit from 0 to 9.

 Model Architecture
The CNN model consists of:
Convolutional layers with ReLU activation
Max-Pooling layers for dimensionality reduction
Fully connected (Dense) layers
Dropout for regularization
Softmax output layer for multi-class classification

 Model Performance
Training Accuracy: ~99%
Validation Accuracy: ~98–99%

Loss Function: Categorical Cross-Entropy

Optimizer: Adam

The model shows strong generalization on unseen validation data.
