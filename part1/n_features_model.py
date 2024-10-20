from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,  LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

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

scorer = make_scorer(compute_rmse,greater_is_better=False)

def find_best_n_features(model, X, y, max_features):
    feature_counts = range(1, max_features + 1)
    scores = []

    for n_features in feature_counts:
        sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward', cv=5)
        sfs.fit(X, y)
        selected_features = sfs.get_support(indices=True)
        
        # Evaluate the model using cross-validation
        X_selected = X[:, selected_features]
        score = cross_val_score(model, X_selected, y, cv=5, scoring=scorer).mean()
        scores.append(-score)    
        print(f"Features: {selected_features}, RMSE: {-score}")
    return feature_counts, scores

def main():
   # Load the data (adjust paths as needed)
    X_train = pd.read_csv(r'..\data_students\labeled_data\X_train.csv')
    y_train = pd.read_csv(r'..\data_students\labeled_data\y_train.csv', header=None).values.ravel()

    ##################
    # preprocesing data
    ###################
    X_train = X_train.drop(columns=['img_filename'])
    

    X_train = map_ordinal_features(X_train)
    
  
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns

    X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())

    X_train_preprocessed, feature_names = preprocess_data(X_train)

    #inti model
    
    model = LinearRegression(learning_rate=0.1, n_iterations=2000)


    # Define the maximum number of features to test
    max_features = 12  # Adjust this based on your dataset

    # Find the best number of features
    feature_counts, scores = find_best_n_features(model, X_train_preprocessed, y_train, max_features)

    # Plot the results
    plt.plot(feature_counts, scores, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Feature Selection: MSE vs Number of Features')
    plt.xticks(feature_counts)  # Set x-ticks to be the feature counts
    plt.show()

    # Find the best number of features
    best_n_features = feature_counts[np.argmin(scores)]
    print(f"Best number of features: {best_n_features}")

if __name__ == "__main__":
    main()
