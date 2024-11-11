c# Machine Learning Model for Compressed Matrix Classification

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

pip install numpy scikit-learn tensorflow

## Data Format

The input data should be in a compressed matrix format saved as a text file. Each row represents a sample with a label followed by feature indices with non-zero values.


- The first value in each row is the label (0 or 1 for binary classification).
- Remaining values are feature indices where the feature is active for that sample.

## Code Structure

- `to_matrix(filename, booli)`: Converts compressed matrix data into a full binary matrix for training and testing.
  
### Model Training:
- **Decision Tree Classifier**: A weighted Decision Tree model, with cross-validation and F1 score as the metric.
- **Artificial Neural Network (ANN)**: A simple neural network using Keras, also cross-validated with F1 score.

### Prediction and Output:
- The trained model predicts labels on test data, outputting predictions as binary values in a text file.

## Usage

1. Place your training and testing data files (`TrainingData.txt` and `TestingData.txt`) in the project directory.
2. Run the script:
   ```bash
   python script_name.py
After execution, predictions will be written to `TestTree.txt` in binary format.

## Output

The script outputs a file named `TestTree.txt`, containing predictions for each row in the test data file:

- `1` for a positive prediction.
- `0` for a negative prediction.

## Notes

- **Class Imbalance**: The Decision Tree classifier uses class weights to handle potential class imbalances.
- **Cross-Validation**: Both models are cross-validated to ensure stable performance metrics.
- **Hyperparameter Tuning**: Modify the class weights, ANN architecture, or number of cross-validation folds to optimize performance for your specific dataset.

