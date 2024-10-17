import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression

def compute_rmse(predict, target):
    diff = predict - np.squeeze(target)  # target is Xw
    return np.sqrt((diff ** 2).sum() / len(target))  # len(target) is P

def gradient_descent(X_train, y_train, learning_rate, num_iterations, lambda_reg=0.1):
    m = X_train.shape[0]  # Number of training samples
    w = np.zeros(X_train.shape[1])  # Initialize weights

    for i in range(num_iterations):
        y_pred_train = X_train.dot(w)
        error = y_pred_train - y_train
        gradient = (1/m) * (X_train.T.dot(error) + lambda_reg * w)  # Add regularization term
        w -= learning_rate * gradient

    return w

def find_best_hyperparams(X_train, y_train, X_val, y_val, learning_rates, num_iterations_list):
    best_rmse = float('inf')
    best_params = {'learning_rate': None, 'num_iterations': None}
    best_weights = None

    for learning_rate in learning_rates:
        for num_iterations in num_iterations_list:
            w = gradient_descent(X_train, y_train, learning_rate, num_iterations)
            y_val_pred = X_val.dot(w)
            val_rmse = compute_rmse(y_val_pred, y_val)

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_params['learning_rate'] = learning_rate
                best_params['num_iterations'] = num_iterations
                best_weights = w

            print(f"Learning Rate: {learning_rate}, Num Iterations: {num_iterations}, Validation RMSE: {val_rmse}")

    return best_params, best_weights

# Load the data (adjust paths as needed)
X_train = pd.read_csv(r'..\data_students\labeled_data\X_train.csv')
X_test = pd.read_csv(r'..\data_students\labeled_data\X_test.csv')
y_train = pd.read_csv(r'..\data_students\labeled_data\y_train.csv', header=None).values.ravel()
y_test = pd.read_csv(r'..\data_students\labeled_data\y_test.csv', header=None).values.ravel()

# Drop the last column containing the image file names from X
X_train = X_train.iloc[:, :-1]
X_test = X_test.iloc[:, :-1]


# Ensure y_train is a pandas Series
y_train_series = pd.Series(y_train)

# Ensure X_train contains only numeric columns
X_train_numeric = X_train.select_dtypes(include=[np.number])

# Calculate mutual information for each feature
mi_scores = mutual_info_regression(X_train_numeric, y_train_series)

# Create a DataFrame for mutual information scores
mi_scores_df = pd.DataFrame({'Feature': X_train_numeric.columns, 'MI Score': mi_scores})

# Select top features based on mutual information scores
top_features = mi_scores_df.sort_values(by='MI Score', ascending=False).head(7)['Feature'].tolist()
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]
print(X_train_top.head())

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

categorical_features = [col for col in X_train_final.columns if X_train[col].dtype == 'object']
numerical_features = [col for col in X_train_final.columns if X_train[col].dtype in [np.number]]


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)


X_train_final_preprocessed = preprocessor.fit_transform(X_train_final)
X_val_preprocessed = preprocessor.transform(X_val)

# add bias term
X_train_final_preprocessed = np.c_[np.ones(X_train_final_preprocessed.shape[0]), X_train_final_preprocessed]
X_val_preprocessed = np.c_[np.ones(X_val_preprocessed.shape[0]), X_val_preprocessed]

# Define hyperparameter grid
learning_rates = [0.0001, 0.001, 0.01]
num_iterations_list = [100, 500, 1000]

# Find the best hyperparameters
best_params, best_weights = find_best_hyperparams(X_train_final_preprocessed, y_train_final, X_val_preprocessed, y_val, learning_rates, num_iterations_list)

print(f"Best Hyperparameters: {best_params}")

# Evaluate the model on the validation set
y_val_pred = X_val_preprocessed.dot(best_weights)
val_rmse = compute_rmse(y_val_pred, y_val)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Validation RMSE: {val_rmse}")
print(f"Validation R^2: {val_r2}")

# Step 4: Evaluate the model on the test set
X_test_preprocessed = preprocessor.transform(X_test)
X_test_preprocessed = np.c_[np.ones(X_test_preprocessed.shape[0]), X_test_preprocessed]
y_test_pred = X_test_preprocessed.dot(best_weights)

# Calculate test metrics
test_rmse = compute_rmse(y_test_pred, y_test)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Test RMSE: {test_rmse}")
print(f"Test R^2: {test_r2}")

# Plot the predictions vs. actual values for the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', linewidth=2)
plt.xlabel("Actual Risk")
plt.ylabel("Predicted Risk")
plt.title("Predicted vs. Actual Risk on Test Set")
plt.grid(True)
plt.show()

# Residual plot for the test set
residuals = y_test - y_test_pred
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals (Test Set)')
plt.grid(True)
plt.show()
