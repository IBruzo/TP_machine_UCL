import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler,  LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0

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

    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate, "n_iterations": self.n_iterations}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def score(self, X, y):
        y_pred = self.predict(X)
        u = np.sum((y - y_pred) ** 2)
        v = np.sum((y - np.mean(y)) ** 2)
        return 1 - (u / v)


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


label_encoder = LabelEncoder()

def preprocess_data(X):
    # Define the preprocessing steps
    numeric_features = ['age', 'blood pressure', 'calcium', 'cholesterol', 
                        'hemoglobin', 'height', 'potassium', 'vitamin D', 'weight','sarsaparilla', 'smurfberry liquor', 'smurfin donuts']
    
    # Transform the categorical 'profession' column using the LabelEncoder
    X['profession'] = label_encoder.fit_transform(X['profession'])
    categorical_features = ['profession']

    # Create a ColumnTransformer for the numerical features
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'  # Keep the transformed 'profession' column as-is
    )

    # Apply the transformations
    X_preprocessed = preprocessor.fit_transform(X)
    feature_names = numeric_features + categorical_features
    return X_preprocessed, feature_names

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


# Create some synthetic data for demonstration
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
    #non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns


    # Handle NaN values in numeric columns
X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].mean())


    # Preprocess the data
X_train_preprocessed, feature_names = preprocess_data(X_train)
X_test_preprocessed, _ = preprocess_data(X_test)



# Define the model
model = LinearRegression()

# Define the parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'n_iterations': [500, 1000, 2000]
}

score = make_scorer(compute_rmse,greater_is_better=False)
# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='score')

# Fit the model
grid_search.fit(X_train_preprocessed, y_train)

# Get the best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Hyperparameters:", best_params)
print("Best Score (RMSE):", best_score)  # Lower is better
