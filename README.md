# Machine Learning Model for Compressed Matrix Classification

This project trains and evaluates multiple machine learning models on data from a compressed matrix format. The code includes functionality for data preprocessing, model training and evaluation, as well as test data prediction. The models include a weighted Decision Tree and an Artificial Neural Network (ANN) classifier.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Data Format](#data-format)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Output](#output)
- [Notes](#notes)

## Overview

The script reads training data in a compressed matrix format, prepares it for modeling, and trains a Decision Tree and ANN model to classify the data. The models are evaluated using cross-validation to assess performance. Finally, the trained model predicts labels on test data, saving the predictions to a file.

## Requirements

- Python 3.x
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
  
Install dependencies via:
```bash
pip install numpy scikit-learn tensorflow
