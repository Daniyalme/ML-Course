# Linear Regression Projects

This folder contains two projects related to machine learning regression models: one focused on predicting song sales and another on predicting house prices. Both projects use various regression techniques to create accurate predictive models.

## 1. Predicting Song Sales

### Overview
This project predicts song sales using a dataset of song-related features. The following regression models are applied:

### Models
- **Linear Regression**: Fits a linear model to the data.
- **Lasso Regression**: Uses L1 regularization for feature selection.
- **Ridge Regression**: Applies L2 regularization to reduce overfitting.

### Key Steps
- Data preprocessing, including scaling and handling missing values.
- Training and evaluating models with metrics such as Mean Squared Error (MSE) and R-squared.

## 2. Predicting House Pricing with Stacked Regression

### Overview
This project predicts house prices by stacking multiple advanced regression models to improve prediction accuracy.

### Models
- **Gradient Boosting**: An ensemble method for sequential model building.
- **Lasso Regression**: For feature selection and regularization.
- **Elastic Net**: Combines Lasso and Ridge for balanced regularization.
- **Kernel Ridge**: Uses kernel functions for non-linear relationships.

### Key Steps
- Preprocess data and train individual models.
- Stack models and evaluate using Root Mean Squared Error (RMSE) and R-squared.

## How to Run
- Ensure you have the necessary libraries installed (`scikit-learn`, `numpy`, `pandas`, `xgboost`).
- Run the corresponding Jupyter Notebook for each project.
