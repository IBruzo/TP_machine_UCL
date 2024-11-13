import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from itertools import product


def preprocess_data(X):

    # Table for features
    numeric_features = ['age', 'blood pressure', 'calcium', 'cholesterol', 'hemoglobin', 'height', 'potassium', 'vitamin D', 'weight','sarsaparilla', 'smurfberry liquor', 'smurfin donuts']
    
    # Table for features with a score
    ordinal_features = ['sarsaparilla', 'smurfberry liquor', 'smurfin donuts']
    feat_score = {"Very low":1, "Low":2, "Moderate":3, "High":4, "Very high":5}
    for feature in ordinal_features:
        X[feature] = X[feature].map(feat_score)

    # Table for professions
    categorical_features = ['profession']
    onehot_encoder = OneHotEncoder(sparse_output=False)  # Ensure dense output to easily combine later
    X_categorical_encoded = onehot_encoder.fit_transform(X[categorical_features])


    # Normalize & scale the combined features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform( X[numeric_features + ordinal_features].values)

    X_combined = np.hstack([
       X_scaled,  
        X_categorical_encoded 
    ])

    # Extract new professions column names
    encoded_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
    feature_names = numeric_features + ordinal_features + list(encoded_feature_names)

    return X_combined, feature_names

def compute_rmse(predict, target):
    diff = predict - np.squeeze(target)
    return np.sqrt((diff ** 2).sum() / len(target))

def corr_filter(corr, n_features, upper_threshold=0.95, lower_threshold=0.1, tighten=0.01, max_iterations=100):
        corr_copy = corr.copy()
        filtered_corr = corr_copy[(corr_copy.abs() >= lower_threshold) & (corr_copy.abs() <= upper_threshold)]
        
        iteration = 0
        while len(filtered_corr) > n_features and iteration < max_iterations:
            lower_threshold += tighten
            upper_threshold -= tighten
            filtered_corr = corr_copy[(corr_copy.abs() >= lower_threshold) & (corr_copy.abs() <= upper_threshold)]
            iteration += 1
        
        iteration = 0
        while len(filtered_corr) < n_features and iteration < max_iterations:
            lower_threshold -= tighten
            upper_threshold += tighten
            filtered_corr = corr_copy[(corr_copy.abs() >= lower_threshold) & (corr_copy.abs() <= upper_threshold)]
            iteration += 1
        
        # If it reaches max_iterations without reaching n_features, return the closest result
        if len(filtered_corr) != n_features:
            print(f"Warning: Could not reach exactly n_features within max_iterations {max_iterations}. Selected {len(filtered_corr)} features. {n_features}")
        
        selected_features = filtered_corr.index.tolist()
        return selected_features

# Define the scorer with the RMSE metric used in practical lessons
scorer = make_scorer(compute_rmse,greater_is_better=False)



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
    # Keep one KF for cross validation
    kf= KFold(n_splits= 5,shuffle=True, random_state=42)
    best_params = None
    best_rmse = np.inf

    y_train_series = pd.Series(y_train)
    corr = X_train_preprocessed_df.corrwith(y_train_series)
    
    param_grid = {
    'n_features': [9,10,11, 12,16,18],
    'upper_threshold': [0.5,0.55,0.6,],
    'lower_threshold': [0.001, 0.0001],
    'tighten': [ 0.01, 0.1,0.001],
    'max_iterations': [100,150,200],
    }

    # Define cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)


    # Grid Search
    for n_features, upper, lower, tighten, max_iter in product(param_grid["n_features"],param_grid["upper_threshold"], 
                                                param_grid["lower_threshold"], 
                                                param_grid["tighten"], 
                                                param_grid["max_iterations"]):
        selected_features = corr_filter(corr, n_features, upper, lower, tighten, max_iter)
        
        
        # Cross-validation RMSE
        # Cross-validation RMSE
        rmses = []
        for train_index, val_index in kf.split(X_train_preprocessed_df):
            # Use .iloc to select rows and columns by index and column names
            X_train_cross = X_train_preprocessed_df.iloc[train_index][selected_features]
            X_val = X_train_preprocessed_df.iloc[val_index][selected_features]
            y_train_cross, y_val = y_train[train_index], y_train[val_index]
            
            model = LinearRegression()
            model.fit(X_train_cross, y_train_cross)
            y_pred = model.predict(X_val)
            rmse = compute_rmse(y_val, y_pred)
            rmses.append(rmse)

        
        mean_rmse = np.mean(rmses)
        
        # Update best parameters if the current mean RMSE is lower
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = {
                "n_features": len(selected_features),   
                "upper_threshold": upper,
                "lower_threshold": lower,
                "tighten": tighten,
                "max_iterations": max_iter
            }

    print("Best Parameters:")
    print(best_params)
    print("Best Cross-Validated RMSE:", best_rmse)

if __name__ == "__main__":
    main()
