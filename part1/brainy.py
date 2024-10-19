import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            error = y_predicted - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                loss = (1 / (2 * n_samples)) * np.sum(error ** 2)
                #print(f"Iteration {i}: Loss = {loss}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def preprocess_data(X):
    # Define the preprocessing steps
    numeric_features = ['age', 'blood pressure', 'calcium', 'cholesterol', 'hemoglobin', 'height', 'potassium', 'vitamin D', 'weight']
    categorical_features = ['profession']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply the transformations
    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

def map_ordinal_features(X):
    ordinal_mapping = {
        'Very Low': 1,
        'Low': 2,
        'Moderate': 3,
        'High': 4,
        'Very High': 5
    }
    ordinal_features = ['sarsaparilla', 'smurfberry liquor', 'smurfin donuts']
    for feature in ordinal_features:
        X[feature] = X[feature].map(ordinal_mapping)
    return X

def compute_rmse(predict, target):
    diff = predict - np.squeeze(target)
    return np.sqrt((diff ** 2).sum() / len(target))


def forward_search(X, y, model, k_features):
    n_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))
    best_rmse = float('inf')

    while len(selected_features) < k_features:
        best_feature = None
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_train_subset = X[:, current_features]
            model.fit(X_train_subset, y)
            predictions = model.predict(X_train_subset)
            rmse = compute_rmse(predictions, y)
            if rmse < best_rmse:
                best_rmse = rmse
                best_feature = feature
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        print(f"Selected features: {selected_features}, RMSE: {best_rmse}")

    return selected_features , best_rmse

def main():
    # Load the data (adjust paths as needed)
    X_train = pd.read_csv(r'..\data_students\labeled_data\X_train.csv')
    X_test = pd.read_csv(r'..\data_students\labeled_data\X_test.csv')
    y_train = pd.read_csv(r'..\data_students\labeled_data\y_train.csv', header=None).values.ravel()
    y_test = pd.read_csv(r'..\data_students\labeled_data\y_test.csv', header=None).values.ravel()

    # Drop non-numeric columns
    X_train = X_train.drop(columns=['img_filename'])
    X_test = X_test.drop(columns=['img_filename'])

    # Map ordinal features to numeric values
    X_train = map_ordinal_features(X_train)
    X_test = map_ordinal_features(X_test)

    # Separate numeric and non-numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns

    # Handle NaN values in numeric columns
    X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
    X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].mean())

    # Handle NaN values in non-numeric columns
    X_train[non_numeric_cols] = X_train[non_numeric_cols].fillna(X_train[non_numeric_cols].mode().iloc[0])
    X_test[non_numeric_cols] = X_test[non_numeric_cols].fillna(X_test[non_numeric_cols].mode().iloc[0])

    # Preprocess the data
    X_train_preprocessed, feature_names = preprocess_data(X_train)
    X_test_preprocessed, _ = preprocess_data(X_test)

    # Feature selection using forward search
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    best_features = []
    best_rmse = float('inf')
    for i in range(1, 10):
        selected_features, acum_rmse = forward_search(X_train_preprocessed, y_train, model, k_features=i)
        if acum_rmse < best_rmse:
            best_rmse = acum_rmse
            best_features = selected_features

    best_feature_names = [feature_names[i] for i in best_features]
    print(f"Best features: {best_feature_names}, Best RMSE: {best_rmse}")

    # Select the top features
    X_train_selected = X_train_preprocessed[:, best_features]
    X_test_selected = X_test_preprocessed[:, best_features]

    # Initialize and train the model
    model.fit(X_train_selected, y_train)

    # Make predictions
    predictions = model.predict(X_test_selected)
    #print("Predictions:", predictions)

    # Evaluate the model using RMSE
    rmse = compute_rmse(predictions, y_test)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Visualize the predictions
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.show()

if __name__ == "__main__":
    main()