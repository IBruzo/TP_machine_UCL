import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def compute_rmse(predict, target):
    diff = predict- np.squeeze(target) #target es Xw
    return np.sqrt((diff**2).sum()/len(target)) #len(target) es P

X_train = pd.read_csv(r'..\data_students\labeled_data\X_train.csv')
X_test = pd.read_csv(r'..\data_students\labeled_data\X_test.csv')
y_train = pd.read_csv(r'..\data_students\labeled_data\y_train.csv', header=None).values.ravel()
y_test = pd.read_csv(r'..\data_students\labeled_data\y_test.csv', header=None).values.ravel()

# Drop the last column image file
X_train = X_train.iloc[:, :-1]
X_test = X_test.iloc[:, :-1]


categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Define a preprocessor with OneHotEncoder for categorical features and StandardScaler for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)


X_train_preprocessed = preprocessor.fit_transform(X_train)

X_test_preprocessed = preprocessor.transform(X_test)

# Create a DataFrame for correlation analysis
# Get the names of the one-hot encoded features
one_hot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

# Combine numerical and one-hot feature names
all_feature_names = numerical_features + list(one_hot_feature_names)

# Create a DataFrame for the training set with the correct feature names
X_train_df = pd.DataFrame(X_train_preprocessed, columns=all_feature_names)

# Calculate the correlation with the target variable
correlation_with_target = X_train_df.corrwith(pd.Series(y_train))

# Select the top 7 features based on absolute correlation with the target
top_features = correlation_with_target.abs().nlargest(7).index

# Get the indices of the selected features
selected_indices = [X_train_df.columns.get_loc(feature) for feature in top_features]

# Filter training and test sets to include only selected features
X_train_selected = X_train_preprocessed[:, selected_indices]
X_test_selected = X_test_preprocessed[:, selected_indices]

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train_selected, y_train)

# Predictions on the test set
y_pred_test = model.predict(X_test_selected)

test_rmse = compute_rmse(y_pred_test, y_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Test RMSE: {test_rmse}")
print(f"Test R^2: {test_r2}")

# Plotting true vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2)
plt.xlabel("True Risk")
plt.ylabel("Predicted Risk")
plt.title("Predicted vs. Actual Risk on Test Set")
plt.grid(True)
plt.show()

# Residual plot
residuals = y_test - y_pred_test
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals (Test Set)')
plt.grid(True)
plt.show()
