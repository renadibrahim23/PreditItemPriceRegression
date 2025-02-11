#USING SUPPORT VECTOR REGRESSOR



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures,MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
import optuna

# Load the training dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# Split data into features and target variable
X = train_data.drop(columns='Y')
y = train_data['Y']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=48)

# Fill NaN values within groups using backfill and forward fill
X_train['X2'] = X_train.groupby('X1')['X2'].transform(lambda group: group.fillna(method='bfill').fillna(method='ffill'))
X_val['X2'] = X_val.groupby('X1')['X2'].transform(lambda group: group.fillna(method='bfill').fillna(method='ffill'))
test_data['X2'] = test_data.groupby('X1')['X2'].transform(lambda group: group.fillna(method='bfill').fillna(method='ffill'))

# Check for any remaining NaN values and replace them with a fallback (e.g., overall mean of X2)
fallback_value = X_train['X2'].mean()  # Can use median or another statistic if preferred
X_train['X2'].fillna(fallback_value, inplace=True)
X_val['X2'].fillna(fallback_value, inplace=True)
test_data['X2'].fillna(fallback_value, inplace=True)

# X4
X_train['X4'] = (X_train['X4'] * 100).round(0)
X_val['X4'] = (X_val['X4'] * 100).round(0)
test_data['X4'] = (test_data['X4'] * 100).round(0)


'''
# Function to replace 0 values in X4 with the mean of X4 for the corresponding X1 group
def replace_zero_with_group_mean(df):
    group_means = df.groupby('X1')['X4'].transform('mean')
    df.loc[df['X4'] == 0, 'X4'] = group_means[df['X4'] == 0]
    return df


X_train = replace_zero_with_group_mean(X_train)
X_val = replace_zero_with_group_mean(X_val)
test_data = replace_zero_with_group_mean(test_data)
'''
# Calculate group means for each dataset

train_group_means = X_train.groupby('X1')['X6'].transform('mean')
val_group_means = X_val.groupby('X1')['X6'].transform('mean')
test_group_means = test_data.groupby('X1')['X6'].transform('mean')

# Replace all X6 values within each group with the corresponding group mean
X_train['X6'] = train_group_means
X_val['X6'] = val_group_means
test_data['X6'] = test_group_means

missing_values = X_train.isnull().sum()


X_train['X1_letters'] = X_train['X1'].str[:2]
X_val['X1_letters'] = X_val['X1'].str[:2]
test_data['X1_letters'] = test_data['X1'].str[:2]
X_train['X1_numbers'] = X_train['X1'].str[3:].astype(int)
X_val['X1_numbers'] = X_val['X1'].str[3:].astype(int)
test_data['X1_numbers'] = test_data['X1'].str[3:].astype(int)

X_train['X1_X2']=X_train['X1_numbers']*X_train['X2']
X_val['X1_X2']=X_val['X1_numbers']*X_val['X2']
test_data['X1_X2']=test_data['X1_numbers']*test_data['X2']




#try to categorize X1 numbers
X_train = pd.get_dummies(X_train, columns=['X1_numbers'], drop_first=False)
X_val = pd.get_dummies(X_val, columns=['X1_numbers'], drop_first=False)
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
test_data = pd.get_dummies(test_data, columns=['X1_numbers'], drop_first=False)
test_data, _ = test_data.align(X_train, join='left', axis=1, fill_value=0)


X_train['X6'] = (X_train['X6']).round(3)
X_val['X6'] = (X_val['X6']).round(3)
test_data['X6'] = (test_data['X6']).round(3)

X_train['Price_Per_Unit_Weight'] = X_train['X6'] / X_train['X2']
X_val['Price_Per_Unit_Weight'] = X_val['X6'] / X_val['X2']
test_data['Price_Per_Unit_Weight'] = test_data['X6'] / test_data['X2']





X_train = pd.get_dummies(X_train, columns=['X1_letters'], drop_first=False)
X_val = pd.get_dummies(X_val, columns=['X1_letters'], drop_first=False)
test_data = pd.get_dummies(test_data, columns=['X1_letters'], drop_first=False)
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
test_data, _ = test_data.align(X_train, join='left', axis=1, fill_value=0)

X_train = X_train.drop(columns='X1')
X_val = X_val.drop(columns='X1')
test_data = test_data.drop(columns='X1')


X_train['X4'] = (X_train['X4'] * 100).round(0)
X_val['X4'] = (X_val['X4'] * 100).round(0)
test_data['X4'] = (test_data['X4'] * 100).round(0)



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

X_train = pd.get_dummies(X_train, columns=['X5_Category_Group'], drop_first=False)
X_val = pd.get_dummies(X_val, columns=['X5_Category_Group'], drop_first=False)
test_data = pd.get_dummies(test_data, columns=['X5_Category_Group'], drop_first=False)
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
test_data, _ = test_data.align(X_train, join='left', axis=1, fill_value=0)

X_train = X_train.drop(columns='X5')
X_val = X_val.drop(columns='X5')
test_data = test_data.drop(columns='X5')

X_train = X_train.drop(columns=['X3'])
X_val = X_val.drop(columns=['X3'])
test_data = test_data.drop(columns=['X3'])

# X6
X_train['X6'] = np.log1p(X_train['X6'])
X_val['X6'] = np.log1p(X_val['X6'])
test_data['X6'] = np.log1p(test_data['X6'])




X_train = X_train.drop(columns='X8')
X_val = X_val.drop(columns='X8')
test_data = test_data.drop(columns='X8')

X_train = pd.get_dummies(X_train, columns=['X7'], drop_first=False)
X_val = pd.get_dummies(X_val, columns=['X7'], drop_first=False)
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
test_data = pd.get_dummies(test_data, columns=['X7'], drop_first=False)
test_data, _ = test_data.align(X_train, join='left', axis=1, fill_value=0)


# X9
X_train['X9'] = X_train['X9'].fillna('Small').astype(object)
X_val['X9'] = X_val['X9'].fillna('Small').astype(object)
test_data['X9'] = test_data['X9'].fillna('Small').astype(object)

X_train = pd.get_dummies(X_train, columns=['X9'], drop_first=False)
X_val = pd.get_dummies(X_val, columns=['X9'], drop_first=False)
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
test_data = pd.get_dummies(test_data, columns=['X9'], drop_first=False)
test_data, _ = test_data.align(X_train, join='left', axis=1, fill_value=0)

'''
X_train['X7_X9'] = X_train['X7'] * X_train['X9']
X_val['X7_X9'] = X_val['X7'] * X_val['X9']
test_data['X7_X9'] = test_data['X7'] * test_data['X9']

X_train = X_train.drop(columns=['X9', 'X7'])
X_val = X_val.drop(columns=['X9', 'X7'])
test_data = test_data.drop(columns=['X9', 'X7'])
'''
'''

le10=LabelEncoder()
X_train['X10']=le10.fit_transform(X_train['X10'])
X_val['X10']=le10.transform(X_val['X10'])
test_data['X10']=le10.transform(test_data['X10'])
'''

X_train = pd.get_dummies(X_train, columns=['X10'], drop_first=False)
X_val = pd.get_dummies(X_val, columns=['X10'], drop_first=False)
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
test_data = pd.get_dummies(test_data, columns=['X10'], drop_first=False)
test_data, _ = test_data.align(X_train, join='left', axis=1, fill_value=0)

# Process X11
X_train = pd.get_dummies(X_train, columns=['X11'], drop_first=False)
X_val = pd.get_dummies(X_val, columns=['X11'], drop_first=False)
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
test_data = pd.get_dummies(test_data, columns=['X11'], drop_first=False)
test_data, _ = test_data.align(X_train, join='left', axis=1, fill_value=0)

# Align columns again to ensure consistency
X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
test_data, _ = test_data.align(X_train, join='left', axis=1, fill_value=0)

# Polynomial Features

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)
test_data_poly = poly.transform(test_data)


# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_data_scaled = scaler.transform(test_data)
'''
# Define objective function with Optuna for SVR
def objective_svr(trial):
    params = {
        'C': trial.suggest_loguniform('C', 0.1, 100),
        'epsilon': trial.suggest_loguniform('epsilon', 0.01, 1),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
    }

    model = SVR(**params)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_val_scaled)
    mae = mean_absolute_error(y_val, preds)
    return mae

# OPTUNA Tuning
study_svr = optuna.create_study(direction="minimize")
study_svr.optimize(objective_svr, n_trials=100)
print("Best SVR Params:", study_svr.best_params)
best_svr = SVR(**study_svr.best_params)
'''
 # # Use the best parameters found in trial 19
best_params =  {'C': 0.14805703125398822, 'epsilon': 0.015634198097106422, 'kernel': 'linear'}
best_svr = SVR(**best_params)
best_svr.fit(X_train_scaled, y_train)

# Evaluate the model on the validation set
'''
val_preds = best_svr.predict(X_val_scaled)
val_mae = mean_absolute_error(y_val, val_preds)
print("Validation MAE:", val_mae)
'''
# Train the final model on the combined training and validation data
X_combined = np.vstack((X_train_scaled, X_val_scaled))
y_combined = np.hstack((y_train, y_val))
best_svr.fit(X_combined, y_combined)

# Create submission file
submission = pd.DataFrame({
    'row_id': range(len(test_data_scaled)),
    'Y': best_svr.predict(test_data_scaled)
})
submission.to_csv('SVR_submission10.csv', index=False)

print("Submission file created successfully!")

#  Trial 53 finished with value: 0.37928391208702694 and parameters: {'C': 3.0324197316539396, 'epsilon': 0.032385104352295274, 'kernel': 'linear'}. Best is trial 53 with value: 0.37928391208702694.

# this i got after categorizing both of X6 and X8
# Best is trial 72 with value: 0.4001122866561832.Best SVR Params: {'C': 1.7655661538419507, 'epsilon': 0.2841439484886996, 'kernel': 'rbf'}


#without rounding X6 
#Best is trial 70 with value: 0.37927929621084017.
#Best SVR Params: {'C': 1.0650963779875169, 'epsilon': 0.03290393175675691, 'kernel': 'linear'}


#without rounding X6 and removed the interaction between X7 and X9 and turned them back to one hot encoding
#Best is trial 41 with value: 0.37818696912670824.
#Best SVR Params: {'C': 66.19464418586152, 'epsilon': 0.02559687348142837, 'kernel': 'linear'}


#interaction feature between X1 numbers and X2 with rounding of X6
#Best is trial 36 with value: 0.3781749603350606.
#Best SVR Params: {'C': 0.20723043561631718, 'epsilon': 0.020493150458110135, 'kernel': 'linear'}


#filling X6 with mean 
# Best is trial 81 with value: 0.37746432623537685.
#Best SVR Params: {'C': 0.7200843120876633, 'epsilon': 0.0173017971530006, 'kernel': 'linear'}

# Best is trial 52 with value: 0.3774337630943474.
#Best SVR Params: {'C': 0.14805703125398822, 'epsilon': 0.015634198097106422, 'kernel': 'linear'}
