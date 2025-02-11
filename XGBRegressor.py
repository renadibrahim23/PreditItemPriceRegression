import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.stats import boxcox

# Load the training dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Split data into features and target variable
X = train_data.drop(columns='Y')
y = train_data['Y']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=48)

# Handling Missing Values & Preprocessing

label_encoder = LabelEncoder()
X_train['X1'] = X_train['X1'].str[:2]
X_val['X1'] = X_val['X1'].str[:2]
test_data['X1'] = test_data['X1'].str[:2]

X_train['X1'] = label_encoder.fit_transform(X_train['X1'])
X_val['X1'] = label_encoder.transform(X_val['X1'])
test_data['X1'] = label_encoder.fit_transform(test_data['X1'])

imputer = SimpleImputer(strategy='median')
X_train['X2'] = imputer.fit_transform(X_train[['X2']])
X_val['X2'] = imputer.transform(X_val[['X2']])
test_data['X2'] = imputer.transform(test_data[['X2']])





X_train = X_train.drop(columns=[ 'X3'])
X_val = X_val.drop(columns=[ 'X3'])
test_data = test_data.drop(columns=['X3'])
    
def categorize_x5(value):
        food_items = ['Dairy', 'Meat', 'Fruits and Vegetables', 'Baking Goods', 'Snack Foods',
                      'Frozen Foods', 'Breakfast', 'Canned', 'Breads']
        drinks_and_dairy = ['Soft Drinks', 'Hard Drinks']
        others = ['Household', 'Others', 'Health and Hygiene']

        if value in food_items:
            return 'Food Items'
        elif value in drinks_and_dairy:
            return 'Drinks and Dairy'
        elif value in others:
            return 'Other'
        else:
            return 'Unknown'

X_train['X5_Category_Group'] = X_train['X5'].apply(categorize_x5)
X_val['X5_Category_Group'] = X_val['X5'].apply(categorize_x5)
test_data['X5_Category_Group'] = test_data['X5'].apply(categorize_x5)



X_train = X_train.drop(columns='X5')
X_val = X_val.drop(columns='X5')
test_data = test_data.drop(columns='X5')


X_train = pd.get_dummies(X_train, columns=['X5_Category_Group'], drop_first=False)
X_val = pd.get_dummies(X_val, columns=['X5_Category_Group'], drop_first=False)
test_data = pd.get_dummies(test_data, columns=['X5_Category_Group'], drop_first=False)
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
test_data, _ = test_data.align(X_train, join='left', axis=1, fill_value=0)

X_train = X_train.drop(columns='X5_Category_Group_Other')
X_val = X_val.drop(columns='X5_Category_Group_Other')
test_data = test_data.drop(columns='X5_Category_Group_Other')



le7 = LabelEncoder()
X_train['X7'] = le7.fit_transform(X_train['X7'])
X_val['X7'] = le7.transform(X_val['X7'])
test_data['X7'] = le7.transform(test_data['X7'])

X_train['X9'] = X_train['X9'].fillna('Small').astype(object)
X_val['X9'] = X_val['X9'].fillna('Small').astype(object)
test_data['X9'] = test_data['X9'].fillna('Small').astype(object)

le = LabelEncoder()
X_train['X9'] = le.fit_transform(X_train['X9'])
X_val['X9'] = le.transform(X_val['X9'])
test_data['X9'] = le.transform(test_data['X9'])

X_train['X7_X9'] = X_train['X7'] * X_train['X9']
X_val['X7_X9'] = X_val['X7'] * X_val['X9']
test_data['X7_X9'] = test_data['X7'] * test_data['X9']

X_train = X_train.drop(columns=['X9', 'X7','X10'])
X_val = X_val.drop(columns=[ 'X9','X7','X10'])
test_data = test_data.drop(columns=['X9','X7','X10'])
# X10_encoder = LabelEncoder()
# X_train['X10'] = X10_encoder.fit_transform(X_train['X10'])
# X_val['X10'] = X10_encoder.transform(X_val['X10'])
# test_data['X10'] = X10_encoder.transform(test_data['X10'])


# Process X11
X_train = pd.get_dummies(X_train, columns=['X11'], drop_first=False)
X_val = pd.get_dummies(X_val, columns=['X11'], drop_first=False)
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
test_data = pd.get_dummies(test_data, columns=['X11'], drop_first=False)
test_data, _ = test_data.align(X_train, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)

import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
import numpy as np

def objective(trial):
    # Define parameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    # Train model
    model = xgb.XGBRegressor(
        **params,
        random_state=42,
        objective="reg:squarederror"
    )
    scores = cross_val_score(model, X_train, y_train, scoring="neg_mean_absolute_error", cv=5)
    return -1 * scores.mean()  # Optuna minimizes the objective

# Create and run the Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, timeout=600)  # Adjust n_trials and timeout as needed

# Display the best parameters
print("Best parameters:", study.best_params)

# Train a model with the best parameters
best_params = study.best_params
model = xgb.XGBRegressor(**best_params, random_state=42, objective="reg:squarederror")
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print("Test set MAE:", mae)





'''
print(f"\nValidation MAE after tuning with Optuna: {validation_mae:.4f}")
# Combine train and validation data for final training
X_combined = np.vstack((X_train_scaled, X_val_scaled))
y_combined = np.hstack((y_train, y_val))

# Train stacked model on combined data
best_model.fit(X_combined, y_combined)

# Predict on the test set using the best model
test_predictions = best_model.predict(test_data)

# Create submission file
submission = pd.DataFrame({
    'row_id': range(len(test_data)),  
    'Y': best_model.predict(test_data)
})
submission.to_csv('sunday_submission3.csv', index=False)

print("Submission file created successfully!")
'''


#Best parameters: {'n_estimators': 923, 'max_depth': 3, 'learning_rate': 0.04091995219231979, 'subsample': 0.670547227412622, 'colsample_bytree': 0.9400016548319, 'reg_alpha': 0.5883371539697606, 'reg_lambda': 0.6924233069538993, 'gamma': 1.6875013447736613, 'min_child_weight': 6}

