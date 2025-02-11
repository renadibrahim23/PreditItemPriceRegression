# PreditItemPriceRegression
Machine Learning Regression Models trained on Predict Item Price Dataset from Kaggle

Machine Learning Kaggle Competition

Overview

This repository contains our team's solution for the Kaggle competition. The goal of the competition was to develop a predictive model using machine learning techniques to minimize the Mean Absolute Error (MAE) on the test dataset.

Team

Team Name: Gru and The Minions

Members: Renad Ibrahim, Salma Moussa, Nada  Medhat

Dataset:

The dataset consists of multiple features, including categorical and numerical variables. Some key preprocessing steps were required to clean and prepare the data before training the models.

Approach:

Our approach involved the following key steps:

1-Data Preprocessing

Handling missing values using KNN imputation.

Encoding categorical variables using target encoding for selected features.

Managing outliers in skewed numerical variables.

2-Feature Engineering

Derived new features based on domain knowledge.

Created categorical groupings for certain variables.

3-Model Training & Selection

Implemented and tested multiple regression models:

Linear Regression

Ridge Regression

Lasso Regression

Gradient Boosting Models (e.g., XGBoost, LightGBM)

Support Vector Regressor

4-Evaluated models using cross-validation to ensure generalization.

5-Hyperparameter Tuning

Used grid search and Optuna to find optimal parameters for models.

Final Submission

Combined the best models using ensemble techniques to improve performance.

Submitted predictions and monitored leaderboard performance.

Results

Achieved a final MAE of 0.398, placing 8th on the competition leaderboard.



