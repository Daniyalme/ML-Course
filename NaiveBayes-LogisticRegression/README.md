# Classification Projects using Naive Bayes and Logistic Regression

This folder contains three machine learning classification projects that apply Naive Bayes and Logistic Regression algorithms to solve different tasks. The projects involve hate speech detection, football match result prediction, and image classification of room cleanliness.

---

Here’s a breakdown for each file:

## 1. Hate Speech Detection using Naive Bayes

### Overview
This project focuses on classifying tweets as either racist or non-racist using a Naive Bayes classifier. The NLTK library is used for preprocessing the tweet text. The model is trained on labeled tweets where "1" indicates hate speech and "0" indicates non-hate speech.

### Key Implementation Steps
- Load and preprocess the dataset (tokenization, stop word removal, etc.) using the NLTK library.
- Train a Naive Bayes classifier to classify tweets as hate speech or not.
- Evaluate the model using performance metrics such as precision, recall, and F1-score.

---

## 2. Football Match Result Prediction using Gaussian Naive Bayes

### Overview
This project predicts football match outcomes (win, loss, or draw) based on historical match data using Gaussian Naive Bayes. Various feature engineering techniques are applied to improve predictive performance.

### Key Implementation Steps
- Load the match dataset and prepare the target column (home team result).
- Apply feature engineering to enhance the dataset (removal of irrelevant features, creation of new features).
- Train and evaluate the Gaussian Naive Bayes model.
- Compare performance using different evaluation metrics like accuracy.

---

## 3. Room Cleanliness Classification using Logistic Regression

### Overview
This project classifies images of rooms as either clean or messy using Logistic Regression. Images are resized, normalized, and then classified based on their cleanliness status.

### Key Implementation Steps
- Preprocess the images (resizing, normalization).
- Train a Logistic Regression model on the preprocessed images.
- Evaluate the model's performance using metrics such as accuracy and the confusion matrix.
- Implement a probability threshold to make final predictions.

---

## How to Run
- Ensure you have the necessary libraries installed (`scikit-learn`, `numpy`, `pandas`, `nltk`, `cv2`).
- Run the corresponding Jupyter Notebook for each project.

---
