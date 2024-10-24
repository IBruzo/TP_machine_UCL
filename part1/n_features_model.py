from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.feature_selection import mutual_info_regression as mutual_info
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


def preprocess_data(X):
    numeric_features = ['age', 'blood pressure', 'calcium', 'cholesterol', 'hemoglobin', 'height', 'potassium', 'vitamin D', 'weight','sarsaparilla', 'smurfberry liquor', 'smurfin donuts']
    categorical_features = ['profession']
    ordinal_features = ['sarsaparilla', 'smurfberry liquor', 'smurfin donuts']

    feat_score = {"Very low":1, "Low":2, "Moderate":3, "High":4, "Very high":5}
    for feature in ordinal_features:
        X[feature] = X[feature].map(feat_score)

   
    onehot_encoder = OneHotEncoder(sparse_output=False)  # Ensure dense output to easily combine later
    X_categorical_encoded = onehot_encoder.fit_transform(X[categorical_features])

    
    X_combined = np.hstack([
        X[numeric_features + ordinal_features].values,  
        X_categorical_encoded 
    ])

    #Scale the combined features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    encoded_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
    feature_names = numeric_features + ordinal_features + list(encoded_feature_names)

    return X_scaled, feature_names

def compute_rmse(predict, target):
    diff = predict - np.squeeze(target)
    return np.sqrt((diff ** 2).sum() / len(target))

def mi_filter(mi,n_features):
        mi_copy = mi.copy()
        sorted_mi = mi_copy.abs().sort_values(ascending=False)
        selected_features = sorted_mi.index[:n_features].tolist()
        return selected_features

scorer = make_scorer(compute_rmse,greater_is_better=False)
kf= KFold(n_splits= 5)

def find_best_n_features(model, X, y, max_features,mi,feature_names):
    feature_counts = range(1, max_features + 1)
    scores = []

    for n_features in feature_counts:

        selected_features =  mi_filter(mi,n_features) 
        
        # Evaluate the model using cross-validation
        selected_indices = [feature_names.index(feature) for feature in selected_features]
        X_selected = X[:, selected_indices]
        score = cross_val_score(model, X_selected, y, cv=kf, scoring=scorer).mean()
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
   
    
    X_train_preprocessed, feature_names = preprocess_data(X_train)
   
    print(feature_names)
    #inti model
    
    model = LinearRegression()

    ##################
    # feature selection  mutual info and forward search
    ###################
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
    mi = pd.Series(mutual_info(X_train_preprocessed_df, y_train), index=X_train_preprocessed_df.columns)
    print(mi)

    max_features = 18

    # Find the best number of features
    feature_counts, scores = find_best_n_features(model, X_train_preprocessed, y_train, max_features,mi,feature_names)
    best_n_features = feature_counts[np.argmin(scores)]
    print(f"Best number of features: {best_n_features}")

    # Plot the results
    plt.plot(feature_counts, scores, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Feature Selection: RMSE vs Number of Features')
    plt.xticks(feature_counts)  
    plt.show()

if __name__ == "__main__":
    main()
