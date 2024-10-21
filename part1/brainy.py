import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,  LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import mutual_info_regression as mutual_info
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class LinearRegression:
    def __init__(self, learning_rate, n_iterations):
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
        return -rmse # Return negative RMSE



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


    X_train_preprocessed, feature_names = preprocess_data(X_train)
    X_test_preprocessed, _ = preprocess_data(X_test)
    print(X_train_preprocessed[:3])
    print(X_train_preprocessed.shape)
    #inti model
    print(len(feature_names))
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)

    ##################
    # feature selection  mutual info and forward search
    ###################
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
    print(X_train_preprocessed_df[:3])
    mi = pd.Series(mutual_info(X_train_preprocessed_df, y_train), index=X_train_preprocessed_df.columns)
    print(mi)

    n_features = 13

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

    selected_features= forward_feature_selection(model, X_train_preprocessed, y_train, 6)

    # Select the top features
    X_train_selected = X_train_preprocessed[:, selected_features]
    X_test_selected = X_test_preprocessed[:, selected_features]

    
    model.fit(X_train_selected, y_train)

    predictions = model.predict(X_test_selected)

    # Evaluate the model using RMSE
    rmse = compute_rmse(predictions, y_test)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    # Evaluate the model using R² score
    r2 = r2_score(y_test, predictions)
    print(f"R² Score: {r2}")

    # Visualize the predictions
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    
    slope, intercept = np.polyfit(y_test, predictions, 1)
    y_line = slope * y_test + intercept
    plt.plot(y_test, y_line, color='red', label='Regression Line')

    plt.show()

    # Load new data for predictions
    new_data = pd.read_csv(r'..\data_students\unlabeled_data\X.csv')

    new_data = new_data.drop(columns=['img_filename'])
    # Preprocess the new data
   
    new_data_preprocessed, _ = preprocess_data(new_data)

    
    new_data_selected = new_data_preprocessed[:, selected_features]
    new_predictions = model.predict(new_data_selected)
    pd.DataFrame(new_predictions).to_csv(r'y_pred.csv', index=False, header=False, float_format='%.10f')
    print("Predictions written!")

if __name__ == "__main__":
    main()