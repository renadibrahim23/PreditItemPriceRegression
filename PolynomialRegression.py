import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import boxcox
from sklearn.metrics import mean_absolute_error

# Load the data
train_data = pd.read_csv('train.csv')

# Split data into features and target variable
X = train_data.drop(columns='Y')
y = train_data['Y']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=48)

# X1: Encoding categorical column (first two characters)



X_train['X1'] = X_train['X1'].str[:2]
X_val['X1'] = X_val['X1'].str[:2]

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()

X_train['X1']=label.fit_transform(X_train['X1'])
X_val['X1']=label.transform(X_val['X1'])

'''
X_train_encoded_df = pd.get_dummies(X_train, columns=['X1'], drop_first=False)
X_val_encoded_df = pd.get_dummies(X_val, columns=['X1'], drop_first=False)
X_train, X_val = X_train_encoded_df.align(X_val_encoded_df, join='left', axis=1, fill_value=0)
'''
# X2: Imputation for numerical column
X2_imputer = SimpleImputer(strategy='median')
X_train['X2'] = X2_imputer.fit_transform(X_train[['X2']])
X_val['X2'] = X2_imputer.transform(X_val[['X2']])

# X3: Drop the column
X_train=X_train.drop(columns='X3')
X_val=X_val.drop(columns='X3')


# X4: Box-Cox transformation

X_train['X4'], _ = boxcox(X_train['X4'] + 1)
X_val['X4'], _ = boxcox(X_val['X4'] + 1)

X_train['X4']=X_train['X4'].replace('0',X_train['X4'].mean())
X_val['X4']=X_val['X4'].replace('0',X_val['X4'].mean())


# X5: Encoding categorical column
'''
le=LabelEncoder()
X_train['X5']=le.fit_transform(X_train['X5'])
X_val['X5']=le.transform(X_val['X5'])
'''
import pandas as pd

# Assuming your dataframe is df and the column with these unique values is 'X5'
def categorize_x5(value):
    # Define the groups
    food_items = ['Dairy', 'Meat', 'Fruits and Vegetables', 'Baking Goods', 'Snack Foods', 
                  'Frozen Foods', 'Breakfast', 'Canned', 'Breads', 
                  'Starchy Foods', 'Seafood']
    drinks_and_dairy = ['Soft Drinks', 'Hard Drinks']
    others = ['Household', 'Others','Health and Hygiene']
    
    # Categorize based on the value
    if value in food_items:
        return 'Food Items'
    elif value in drinks_and_dairy:
        return 'Drinks and Dairy'
    elif value in others:
        return 'Other'
    else:
        return 'Unknown'

# Apply the function to 'X5' column to create a new feature
X_train['X5_Category_Group'] = X_train['X5'].apply(categorize_x5)
X_val['X5_Category_Group']=X_val['X5'].apply(categorize_x5)
# View the updated dataframe
print(X_train[['X5', 'X5_Category_Group']].head())

X_train=X_train.drop(columns='X5')
#X_val=X_val.drop(columns='X5')



X_train_encoded_df = pd.get_dummies(X_train, columns=['X5_Category_Group'], drop_first=False)
X_val_encoded_df = pd.get_dummies(X_val, columns=['X5_Category_Group'], drop_first=False)
X_train, X_val = X_train_encoded_df.align(X_val_encoded_df, join='left', axis=1, fill_value=0)


'''
X_train_encoded_df = pd.get_dummies(X_train, columns=['X5'], drop_first=False)
X_val_encoded_df = pd.get_dummies(X_val, columns=['X5'], drop_first=False)
X_train, X_val = X_train_encoded_df.align(X_val_encoded_df, join='left', axis=1, fill_value=0)
'''

'''
X_train['price_per_weight']=X_train['X6']/X_train['X2']
X_val['price_per_weight']=X_val['X6']/X_val['X2']
'''
'''
X_train=X_train.drop(columns='X2')
X_val=X_val.drop(columns='X2')
'''


# X7: Encoding categorical column
from sklearn.preprocessing import LabelEncoder
le3=LabelEncoder()
X_train['X7']=le3.fit_transform(X_train['X7'])
X_val['X7']=le3.transform(X_val['X7'])
'''
X_train_encoded_df = pd.get_dummies(X_train, columns=['X7'], drop_first=False)
X_val_encoded_df = pd.get_dummies(X_val, columns=['X7'], drop_first=False)
X_train, X_val = X_train_encoded_df.align(X_val_encoded_df, join='left', axis=1, fill_value=0)
'''
# X8: Drop the column
X_train = X_train.drop(columns='X8')
X_val = X_val.drop(columns='X8')



# X9: Handle missing values and label encoding
X_train['X9'] = X_train['X9'].fillna('Missing')
X_val['X9'] = X_val['X9'].fillna('Missing')
le = LabelEncoder()
X_train['X9'] = le.fit_transform(X_train['X9'])
X_val['X9'] = le.transform(X_val['X9'])

# X10: Label Encoding
'''
X_train_encoded_df = pd.get_dummies(X_train, columns=['X10'], drop_first=False)
X_val_encoded_df = pd.get_dummies(X_val, columns=['X10'], drop_first=False)
X_train, X_val = X_train_encoded_df.align(X_val_encoded_df, join='left', axis=1, fill_value=0)
'''
X10_encoder = LabelEncoder()
X_train['X10'] = X10_encoder.fit_transform(X_train['X10'])
X_val['X10'] = X10_encoder.transform(X_val['X10'])

# X11: Encoding categorical column
'''
X11_encoder = LabelEncoder()
X_train['X11'] = X11_encoder.fit_transform(X_train['X11'])
X_val['X11'] = X11_encoder.transform(X_val['X11'])
'''
X_train_encoded_df = pd.get_dummies(X_train, columns=['X11'], drop_first=False)
X_val_encoded_df = pd.get_dummies(X_val, columns=['X11'], drop_first=False)
X_train, X_val = X_train_encoded_df.align(X_val_encoded_df, join='left', axis=1, fill_value=0)


from sklearn.preprocessing import MinMaxScaler


# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Training the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100,min_samples_leaf=4,max_depth=10, random_state=42)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import math
rf_model.fit(X_train,y_train)

predictions=rf_model.predict(X_val_scaled)

mae=mean_absolute_error(predictions,y_val)

print(mae)
rand_forest_mse = mean_squared_error(y_val ,predictions)
rand_forest_rmse = math.sqrt(rand_forest_mse)
rand_forest_r2 = r2_score(y_val, predictions)

print('RandomForest RMSE  \t       ----> {}'.format(rand_forest_rmse))
print('RandomForest R2 Score       ----> {}'.format(rand_forest_r2))


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
degree = 2  # You can tune this
poly_pipeline = Pipeline([
    ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Fit the model
poly_pipeline.fit(X_train_scaled, y_train)

# Predict
y_pred = poly_pipeline.predict(X_val_scaled)

# Evaluate the performance
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Absolute Error (MAE) for Polynomial Regression (degree={degree}): {mae}")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_scaled, y_train)

y_train_pred = gb_model.predict(X_train_scaled)
y_val_pred = gb_model.predict(X_val_scaled)

train_mae = mean_absolute_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)

train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Training MAE: {train_mae}, R²: {train_r2}")
print(f"Validation MAE: {val_mae}, R²: {val_r2}")


import matplotlib.pyplot as plt
import numpy as np

# Get the feature importances
importances = gb_model.feature_importances_

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(X_train.shape[1]), importances[indices], align="center")
plt.yticks(range(X_train.shape[1]), X_train.columns[indices])
plt.xlabel("Relative Importance")
plt.show()




#{'colsample_bytree': 0.8, 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 0.8}
