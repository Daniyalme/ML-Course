# Classification Projects using Naive Bayes and Logistic Regression

This folder contains three machine learning classification projects that apply Naive Bayes and Logistic Regression algorithms to solve different tasks. The projects involve hate speech detection, football match result prediction, and image classification of room cleanliness.

---

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

### Dataset
You can download the football match dataset [here](https://drive.google.com/file/d/1X1OJhK5w3WYXFSxJ2A7i4twKIGLnRAg8/view?usp=sharing).

### Dataset Setup
After downloading the dataset, Extract the downloader dataset into the same directory as the project.

**Double-check the code**: Open the code file and verify that the file path (`'match.csv'`) corresponds to the location where the dataset is saved.

### Key Implementation Steps
- Load the match dataset and prepare the target column (home team result).
- Apply feature engineering to enhance the dataset (removal of irrelevant features, creation of new features).
- Train and evaluate the Gaussian Naive Bayes model.
- Compare performance using different evaluation metrics like accuracy.

---

## 3. Room Cleanliness Classification using Logistic Regression

### Overview
This project classifies images of rooms as either clean or messy using Logistic Regression. Images are resized, normalized, and then classified based on their cleanliness status.

### Dataset
You can download the room cleanliness dataset [here](https://drive.google.com/file/d/1rZPfEx2gJLV9neDAvWM1u52NsMBiKld6/view?usp=sharing).

### Dataset Setup
After downloading the dataset, Extract the downloader dataset into the same directory as the project.

**Double-check the code**: Open the code file and verify that the paths correspond to the location where the dataset is saved.

### Key Implementation Steps
- Preprocess the images (resizing, normalization).
- Train a Logistic Regression model on the preprocessed images.
- Evaluate the model's performance using metrics such as accuracy and the confusion matrix.
- Implement a probability threshold to make final predictions.

---

## General Guide to Setting Up Datasets

1. **Create a `data` or `images` folder**: Based on the project (for CSV files or image files, respectively).
2. **Place the downloaded dataset in the correct folder**: Ensure the dataset is saved in the expected location.
3. **Check the file paths in the code**: Open the relevant Python files and confirm that the file paths used in the code (e.g., `'data/match.csv'` or `'images/room_image.jpg'`) match the location and filenames of the downloaded files. If necessary, update the paths to reflect the correct locations.

---

This README now includes a guide for downloading the datasets, placing them into the project folder, and verifying file paths in the code to ensure everything runs smoothly. Replace the placeholder URLs with the actual URLs for the datasets.

Let me know if you need any further changes!
