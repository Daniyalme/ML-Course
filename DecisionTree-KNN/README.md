# Decision Tree and KNN Projects

This folder contains four Machine Learning classification projects that demonstrate the use of K-Nearest Neighbors (KNN) and Decision Trees for various predictive tasks. The projects range from visualizing decision boundaries, and optimizing feature weights, to solving real-world problems such as stroke prediction and car safety classification.

## 1. KNN Classifier Decision Boundary

### Overview
This project visualizes the decision boundary of a K-Nearest Neighbors (KNN) classifier. Decision boundaries help illustrate how the classifier divides the feature space to make predictions.

### Key Steps
- Training a KNN classifier.
- Visualizing the decision boundary for various configurations of the KNN model.
- Displaying how the classifier behaves with different hyperparameters, such as the number of neighbors (K).

## 2. Predicting Strokes using Decision Trees and KNN

### Overview
This project predicts whether a person is at risk of having a stroke by following a structured process that involves data preprocessing, model training, and evaluation using both K-Nearest Neighbors and Decision Trees.

### Key Steps
- **Preprocessing**: Data preprocessing includes handling missing values, normalization, and bias identification.
- **Stratified Sampling**: The dataset is split into training, validation, and test sets using stratified sampling to ensure balanced target classes.
- **Modeling**: A K-Nearest Neighbors (KNN) model and a decision tree model are trained and optimized. Hyperparameters are tuned using methods like the elbow method for KNN and grid search for the decision tree.
- **Model Comparison and Evaluation**: Compares the models and evaluates them using accuracy, F1-score, and other metrics on the test set.
- **Handling Imbalanced Data**: Discusses and applies methods to manage imbalanced data.

## 3. KNN Feature Selection and Weighted Classification

### Overview
This project implements a K-Nearest Neighbors (KNN) classifier with feature selection and weight optimization to classify an insurance dataset. The goal is to calculate optimal feature weights and evaluate the model's performance using these weights.

### Key Steps

- **Dataset Preprocessing**: Load the dataset, perform necessary preprocessing steps, split it into training, validation, and test sets with a ratio of 80/10/10.

- **Weighted Distance Calculation**:
  - Implement a function to calculate the Euclidean distance between each observation and all other data points.
  - Modify the distance calculation to include weights for each feature, ensuring the weighted distance is used for classification.

- **Weight Optimization**:
  - Initialize the feature weights with zeros.
  - Use an optimization function (`scipy.optimize.minimize`) to find the best weights that minimize classification loss based on the training set.

- **KNN Classification with Optimized Weights**:
  - Implement the weighted KNN classifier using the optimized weights.
  - Classify the test set and evaluate performance metrics (accuracy, AUC, etc.).

- **Feature Subset Selection**:
  - Reduce the feature set to smaller subsets (e.g., 5 features).
  - Retrain the KNN model on these subsets, classify the test set, and compare results.

- **Final Feature Selection**:
  - Select the 5 features with the highest weights, retrain the KNN model, and classify the test set.
  - Compare performance with a random selection of features.

This version focuses purely on the steps involved in the implementation. Let me know if it needs further refinement!

## 4. Car Safety Evaluation Using Decision Trees

### Overview
This project evaluates car safety by using a decision tree classifier on the "Car Safety" dataset. The goal is to predict the safety category of a car based on features such as price, maintenance cost, and seating capacity. Both vanilla decision trees and pruning methods are employed to analyze feature importance and improve model accuracy.

### Dataset
The dataset consists of the following features:
- **Buying price ('buying')**
- **Maintenance cost ('maint')**
- **Number of doors ('doors')**
- **Seating capacity ('persons')**
- **Trunk size ('lug_boot')**
- **Overall safety rating ('safety')**

The target variable is the safety class ('class'), which categorizes cars into different safety levels.

### Key Steps
- **Training Decision Trees**: A vanilla decision tree model is trained first, followed by an evaluation using pruned decision trees to improve generalizability.
- **Feature Importance Analysis**: The model is used to assess the importance of different car features in determining the safety classification.
- **Model Evaluation**: The model’s performance is measured using accuracy, precision, and other relevant metrics.

## How to Run
- Ensure the necessary libraries are installed (`scikit-learn`, `numpy`, `pandas`, `scipy`).
- Run the scripts for each project to train the models and view results.

---
