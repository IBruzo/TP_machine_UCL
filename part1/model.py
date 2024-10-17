import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
X_train = pd.read_csv(r'..\data_students\labeled_data\X_train.csv')  # Adjust paths as needed
X_test = pd.read_csv(r'..\data_students\labeled_data\X_test.csv')
y_train = pd.read_csv(r'..\data_students\labeled_data\y_train.csv', header=None).values.ravel()
y_test = pd.read_csv(r'..\data_students\labeled_data\y_test.csv', header=None).values.ravel()

# Drop the last column containing the image file names from X
X_train = X_train.iloc[:, :-1]
X_test = X_test.iloc[:, :-1]

# Identify categorical and numerical features
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Define a preprocessor with OneHotEncoder for categorical features and StandardScaler for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Apply the preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Feature Selection: For simplicity, let's select all features (can be refined later)
X_train_selected = X_train_preprocessed
X_test_selected = X_test_preprocessed

# Convert to NumPy arrays
X_train_selected = np.array(X_train_selected)
X_test_selected = np.array(X_test_selected)

# Add a column of ones to X_train and X_test for the bias term
X_train_selected = np.c_[np.ones(X_train_selected.shape[0]), X_train_selected]
X_test_selected = np.c_[np.ones(X_test_selected.shape[0]), X_test_selected]

# Initialize weights (w) to zeros, including the bias term
w = np.zeros(X_train_selected.shape[1])

# Define gradient descent parameters
learning_rate = 0.01
num_iterations = 1000
m = X_train_selected.shape[0]  # Number of training samples

# Gradient Descent for Linear Regression
for i in range(num_iterations):
    # Compute predictions
    y_pred_train = X_train_selected.dot(w)
    
    # Compute the error
    error = y_pred_train - y_train
    
    # Compute gradients
    gradient = (1/m) * X_train_selected.T.dot(error)
    
    # Update weights
    w -= learning_rate * gradient
    
    # Optionally, print the loss every 100 iterations to monitor progress
    if i % 100 == 0:
        loss = (1/(2*m)) * np.sum(error ** 2)
        print(f"Iteration {i}: Loss = {loss}")

# Evaluate the model on the test set
y_pred_test = X_test_selected.dot(w)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
test_r2 = r2_score(y_test, y_pred_test)

print("Test RMSE:", test_rmse)
print("Test R^2:", test_r2)

# Visualize the results
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2)
plt.xlabel("True Risk")
plt.ylabel("Predicted Risk")
plt.title("True vs. Predicted Risk")
plt.grid(True)
plt.show()

# Residual plot
residuals = y_test - y_pred_test
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True)
plt.show()
