import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,  LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import mutual_info_regression as mutual_info
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.1, n_iterations=2000):
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


def backward_greed_search(model, X, y, n_features):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='backward', cv=5)
    sfs.fit(X, y)
    selected_indices = np.where(sfs.get_support())[0]
    return selected_indices

def forward_feature_selection(model, X, y, n_features):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward', cv=5)
    sfs.fit(X, y)
    selected_indices = np.where(sfs.get_support())[0]
    return selected_indices


def main():
    # Load the data (adjust paths as needed)
    X_train = pd.read_csv(r'..\data_students\labeled_data\X_train.csv')
    X_test = pd.read_csv(r'..\data_students\labeled_data\X_test.csv')
    y_train = pd.read_csv(r'..\data_students\labeled_data\y_train.csv', header=None).values.ravel()
    y_test = pd.read_csv(r'..\data_students\labeled_data\y_test.csv', header=None).values.ravel()

    ##################
    # preprocesing data
    ###################
    X_train = X_train.drop(columns=['img_filename'])
    X_test = X_test.drop(columns=['img_filename'])

    
    X_train = map_ordinal_features(X_train)
    X_test = map_ordinal_features(X_test)

  
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns

    X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
    X_test[numeric_cols] = X_test[numeric_cols].fillna(X_test[numeric_cols].mean())


    X_train_preprocessed, feature_names = preprocess_data(X_train)
    X_test_preprocessed, _ = preprocess_data(X_test)

    #inti model
    
    model = LinearRegression(learning_rate=0.1, n_iterations=2000)

    ##################
    # feature selection  mutual info and forward search
    ###################
    mi = pd.Series(mutual_info(X_train_preprocessed, y_train), index=X_train.columns)
    print(mi)

    n_features = 6

    def mi_filter(mi,n_features):
        mi_copy = mi.copy()
        sorted_mi = mi_copy.abs().sort_values(ascending=False)
        selected_features = sorted_mi.index[:n_features].tolist()
        return selected_features
    
    selected_features =  mi_filter(mi,n_features) 


    print("Selected Features:", selected_features)
    selected_indices = [feature_names.index(feature) for feature in selected_features]

    X_train_selected = X_train_preprocessed[:, selected_indices]
    X_test_selected = X_test_preprocessed[:, selected_indices]

    selected_features= backward_greed_search(model, X_train_preprocessed, y_train, n_features)

    # Select the top features
    X_train_selected = X_train_preprocessed[:, selected_features]
    X_test_selected = X_test_preprocessed[:, selected_features]

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
    
    slope, intercept = np.polyfit(y_test, predictions, 1)
    y_line = slope * y_test + intercept
    plt.plot(y_test, y_line, color='red', label='Regression Line')

    plt.show()

if __name__ == "__main__":
    main()