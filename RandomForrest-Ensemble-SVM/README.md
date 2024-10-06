# Random Forest, SVM, and Ensemble Methods Projects

This folder contains four machine learning projects that focus on comparing and evaluating the performance of different algorithms such as Support Vector Machines (SVM), Random Forest, and various ensemble methods. The projects cover tasks like spam classification, voice rehabilitation analysis, and fraud detection, using techniques such as kernel comparison, hyperparameter tuning, and Out-Of-Bag error estimation.

---

## 1. Kernel Comparison on LSVT Voice Rehabilitation Dataset

### Overview
This project compares the performance of different kernels—RBF, Sigmoid, Polynomial, and Linear—applied to a Support Vector Machine (SVM) on the LSVT Voice Rehabilitation dataset. The goal is to identify which kernel performs best for predicting the rehabilitation outcomes of voice patients.

### Key Implementation Steps
- Load the LSVT dataset and preprocess the data.
- Train SVM models with different kernels (RBF, Sigmoid, Polynomial, Linear).
- Compare model performance based on accuracy and other relevant metrics.


---

## 2. Spam Classification using Random Forest

### Overview
This project applies a Random Forest classifier to classify emails as spam or not. It also explores decision tree stability and the use of Out-Of-Bag (OOB) error for model evaluation. The Random Forest algorithm is evaluated by training on the spam dataset.

### Key Implementation Steps
- Load the dataset and explore the classification task.
- Train a Random Forest classifier and refit the tree using random subsets of data to assess stability.
- Calculate and evaluate the Out-Of-Bag (OOB) error to estimate the generalization ability of the model.
- Visualize the trees using `sklearn.tree.plot_tree`.

### Dataset Setup
1. After downloading the dataset, create a folder named `data` inside the project directory.
2. Place the dataset file (`spam.csv`) into the `data` folder.
3. Verify that the file paths in the code match the location of the dataset, such as `data/spam.csv`.

---

## 3. Comparing Ensemble Methods on Fraud Detection Dataset

### Overview
This project compares the performance of different ensemble methods, including XGBoost, Random Forest, and AdaBoost, to detect fraud. It also includes hyperparameter tuning to optimize the models' performance and compares the effectiveness of these methods on the fraud dataset.

### Key Implementation Steps
- Load and preprocess the fraud detection dataset.
- Train XGBoost, Random Forest, and AdaBoost models on the dataset.
- Tune hyperparameters for each model using grid search or random search.
- Compare the models' performance using metrics like accuracy and AUC.

---

## 4. SVM for Binary Classification

### Overview
This project uses a Support Vector Machine (SVM) to predict a binary outcome on a dataset containing 14 features. The focus is on data preprocessing, scaling, and model evaluation. Different SVM techniques and hyperparameters are explored to optimize the binary classification task.


### Key Implementation Steps
- Preprocess the dataset (handle missing values, normalize features, etc.).
- Split the data into training and test sets.
- Train the SVM model and evaluate its performance using metrics such as accuracy and precision.
- Analyze the effect of regularization (C parameter) on the SVM model's performance.

---
