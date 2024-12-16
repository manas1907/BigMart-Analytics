import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the data
train_data = pd.read_csv("/content/train.csv")
test_data = pd.read_csv("/content/test.csv")

# Combine train and test data for uniform preprocessing
data = pd.concat([train_data, test_data], ignore_index=True)

# Handling missing values
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)
data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace=True)

# Feature Engineering
# Create a new feature for Item Visibility mean ratio
data['Item_Visibility_MeanRatio'] = data.groupby('Item_Identifier')['Item_Visibility'].transform(lambda x: x / x.mean())

# Simplify the Item_Fat_Content feature
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
    'low fat': 'Low Fat',
    'LF': 'Low Fat',
    'reg': 'Regular'
})

# Create a new feature for Outlet years
current_year = 2013
data['Outlet_Years'] = current_year - data['Outlet_Establishment_Year']

# Label Encoding for categorical variables
labelencoder = LabelEncoder()
cat_cols = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type']
for col in cat_cols:
    data[col] = labelencoder.fit_transform(data[col])

# One-Hot Encoding for Outlet_Identifier and Item_Type
data = pd.get_dummies(data, columns=['Outlet_Identifier', 'Item_Type'], drop_first=True)

# Splitting data back into train and test
data_train = data[:len(train_data)]
data_test = data[len(train_data):]

X = data_train.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Establishment_Year'])
y = data_train['Item_Outlet_Sales']
X_test = data_test.drop(columns=['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Establishment_Year'])

# Split the train data for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
print(f"Validation Mean Squared Error: {mse}")

# Predicting on test data
test_predictions = model.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    'Item_Identifier': test_data['Item_Identifier'],
    'Outlet_Identifier': test_data['Outlet_Identifier'],
    'Item_Outlet_Sales': test_predictions
})

# Save submission file
submission.to_csv('BigMart_Sales_Prediction.csv', index=False)
print("Submission file created: BigMart_Sales_Prediction.csv")
