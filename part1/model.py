import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler,  LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin


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
        mse = np.mean((y - y_pred) ** 2) 
        rmse = np.sqrt(mse)  
        return -rmse


    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


label_encoder = LabelEncoder()

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}

    def fit(self, X, y=None):
        for column in X.columns:
            le = LabelEncoder()
            le.fit(X[column])
            self.label_encoders[column] = le
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column, le in self.label_encoders.items():
            X_transformed[column] = le.transform(X[column])
        return X_transformed

def preprocess_data(X):
    # Define the preprocessing steps
    numeric_features = ['age', 'blood pressure', 'calcium', 'cholesterol', 'hemoglobin', 'height', 'potassium', 'vitamin D', 'weight']
    categorical_features = ['profession']
    ordinal_features = ['sarsaparilla', 'smurfberry liquor', 'smurfin donuts']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    ordinal_transformer = LabelEncoderTransformer()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', ordinal_transformer, ordinal_features)
        ])

    # Apply the transformations
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Get the feature names after preprocessing
    feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)) + ordinal_features
    
    return X_preprocessed, feature_names

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

    
    # Preprocess the data
X_train_preprocessed, feature_names = preprocess_data(X_train)
X_test_preprocessed, _ = preprocess_data(X_test)


# Define the model
model = LinearRegression()

# Define the parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'n_iterations': [500, 1000, 2000,5000]
}

score = make_scorer(compute_rmse,greater_is_better=False)
kf= KFold(n_splits= 5)
# Set up the GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,cv=kf, scoring=score)

# Fit the model
grid_search.fit(X_train_preprocessed, y_train)

# Get the best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Hyperparameters:", best_params)
print("Best Score (RMSE):", best_score)  # Lower is better
