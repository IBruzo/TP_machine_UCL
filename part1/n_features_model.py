from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,  LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression as mutual_info
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
    # Define the numeric features
    numeric_features = ['age', 'blood pressure', 'calcium', 'cholesterol', 
                        'hemoglobin', 'height', 'potassium', 'vitamin D', 
                        'weight', 'sarsaparilla', 'smurfberry liquor', 
                        'smurfin donuts']
    
    # Define the categorical features (just 'profession')
    categorical_features = ['profession']

    # Create transformers for numeric and categorical data
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False) # OneHotEncoder drops the first category to avoid multicollinearity

    # Create a ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep any other columns as-is
    )

    # Apply the transformations
    X_preprocessed = preprocessor.fit_transform(X)
    
    profession_categories = preprocessor.named_transformers_['cat'].get_feature_names_out(['profession'])
    feature_names = numeric_features + list(profession_categories)
    
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

def mi_filter(mi,n_features):
        mi_copy = mi.copy()
        sorted_mi = mi_copy.abs().sort_values(ascending=False)
        selected_features = sorted_mi.index[:n_features].tolist()
        return selected_features

scorer = make_scorer(compute_rmse,greater_is_better=False)

def find_best_n_features(model, X, y, max_features,mi,feature_names):
    feature_counts = range(1, max_features + 1)
    scores = []

    for n_features in feature_counts:

        selected_features =  mi_filter(mi,n_features) 
        
        # Evaluate the model using cross-validation
        selected_indices = [feature_names.index(feature) for feature in selected_features]
        X_selected = X[:, selected_indices]
        score = cross_val_score(model, X_selected, y, cv=5, scoring=scorer).mean()
        scores.append(-score)    
        print(f"Features: {selected_features}, RMSE: {-score}")
    return feature_counts, scores

def main():

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
   
    print(feature_names)
    #inti model
    
    model = LinearRegression(learning_rate=0.001, n_iterations=5000)

    ##################
    # feature selection  mutual info and forward search
    ###################
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
    mi = pd.Series(mutual_info(X_train_preprocessed_df, y_train), index=X_train_preprocessed_df.columns)
    print(mi)

    max_features = 17

    # Find the best number of features
    feature_counts, scores = find_best_n_features(model, X_train_preprocessed, y_train, max_features,mi,feature_names)

    # Plot the results
    plt.plot(feature_counts, scores, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Feature Selection: RMSE vs Number of Features')
    plt.xticks(feature_counts)  
    plt.show()

    # Find the best number of features
    best_n_features = feature_counts[np.argmin(scores)]
    print(f"Best number of features: {best_n_features}")

if __name__ == "__main__":
    main()
