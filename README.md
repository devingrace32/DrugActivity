README
Project Overview
This project trains two different machine learning models to classify data from a compressed matrix format and outputs predictions on test data. The models used include a weighted Decision Tree classifier and an Artificial Neural Network (ANN). The script also includes cross-validation to evaluate the models' performance.

Dependencies
Python 3.x
NumPy
Scikit-learn
TensorFlow (for Keras)
Statistics (for calculating mean)
To install the dependencies, use:

bash
Copy code
pip install numpy scikit-learn tensorflow
Script Details
Data Preprocessing:

The to_matrix() function reads data from a compressed matrix file, converting it into a full matrix format.
The data should be provided as a text file, with features in a sparse format.
Model Training:

Decision Tree: A weighted Decision Tree classifier is used, applying a higher weight to one class. This model is cross-validated using StratifiedKFold with F1 score as the evaluation metric.
Artificial Neural Network: The ANN model is created using Keras, with a simple feedforward network architecture. The ANN model is also evaluated with cross-validation.
Generating Predictions:

The script predicts labels on test data using the trained Decision Tree model. The output is saved to a file, with predictions converted to binary classes.
Running the Script
Place the training data (TrainingData.txt) and testing data (TestingData.txt) in the script’s directory.
Run the script:
bash
Copy code
python script_name.py
After execution, predictions will be written to TestTree.txt in binary format.
Notes
Ensure that the input data files are formatted as expected.
You can modify the class weights in the Decision Tree if the class imbalance is different.
For best results, tune hyperparameters as needed.
Output
The final output is a file (TestTree.txt) containing predictions for each row in the test set, with 1 or 0 as the predicted labels.
