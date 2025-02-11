import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
train_data = pd.read_csv('train.csv')

# Display dataset info
print("Dataset Information:")
print(train_data.info())

# Handle missing values
train_data['X2'] = train_data['X2'].fillna(train_data['X2'].mean())
train_data['X6'] = train_data['X6'].fillna(train_data['X6'].mean())

# Handle missing values in X9 using median per group of X11
train_data['X9'] = train_data.groupby('X11')['X9'].transform(lambda x: x.fillna(x.median()))

# Drop unnecessary columns
cols_to_drop = ['X1', 'X3', 'X4', 'X10']
train_data = train_data.drop(cols_to_drop, axis=1)

# Apply Target Encoding on categorical columns
categorical_columns = ['X5', 'X7', 'X9', 'X11']
encoder = TargetEncoder()
train_data[categorical_columns] = encoder.fit_transform(train_data[categorical_columns], train_data['Y'])

# Scale the numerical data
numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])

# Separate features and target variable
features = train_data.drop('Y', axis=1)
target = train_data['Y']

# Add polynomial interaction terms (degree 2 for optimization)
poly = PolynomialFeatures(degree=2, include_bias=False)
interaction_features = poly.fit_transform(features[['X6', 'X7']])
interaction_columns = [f'poly_{i}' for i in range(interaction_features.shape[1])]
interaction_df = pd.DataFrame(interaction_features, columns=interaction_columns, index=features.index)
features = pd.concat([features, interaction_df], axis=1)

# Initialize base models
rf_model = RandomForestRegressor(random_state=58)
xgb_model = XGBRegressor(random_state=42, n_jobs=-1)
cat_model = CatBoostRegressor(verbose=0, random_state=42)
lgb_model = lgb.LGBMRegressor(random_state=42)

# Feature selection using RandomForest
feature_selector = SelectFromModel(rf_model.fit(features, target), threshold=0.01)
features_selected = feature_selector.transform(features)

# Stacking regressor with LassoCV as meta-model
stacked_model = StackingRegressor(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('cat', cat_model),
        ('lgb', lgb_model)
    ],
    final_estimator=LassoCV(cv=5, random_state=42),
    cv=5,
    n_jobs=-1
)

# Cross-validation evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(stacked_model, features_selected, target, cv=kf, scoring='neg_mean_absolute_error')
print(f"Mean MAE from Cross-Validation: {-np.mean(scores):.4f}")

# Train the final model on the full dataset
stacked_model.fit(features_selected, target)

