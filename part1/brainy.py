import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression



def preprocess_data(X):

      # Create BMI from height and weight
    X['BMI'] = X['weight'] / ((X['height'] / 100) ** 2)  # Convert height to meters and compute BMI
    
    # Remove height and weight columns
    X = X.drop(columns=['height', 'weight'])
    
    # Updated list of numeric features
    numeric_features = ['age', 'blood pressure', 'calcium', 'cholesterol', 'hemoglobin', 'potassium', 'vitamin D', 'BMI']
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



def load_data():
    # Load the data (adjust paths as needed)
    X_train = pd.read_csv(r'..\data_students\labeled_data\X_train.csv')
    X_test = pd.read_csv(r'..\data_students\labeled_data\X_test.csv')
    y_train = pd.read_csv(r'..\data_students\labeled_data\y_train.csv', header=None).values.ravel()
    y_test = pd.read_csv(r'..\data_students\labeled_data\y_test.csv', header=None).values.ravel()

    return X_train, X_test, y_train, y_test

def prepare_model(X, y_train, feat_names):

    
    def corr_filter(corr, n_features, upper_threshold=0.7, lower_threshold=0.001, tighten=0.01, max_iterations=100):
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
            print("Warning: Could not reach exactly n_features within max_iterations.")
        
        selected_features = filtered_corr.index.tolist()
        return selected_features

    df = pd.DataFrame(X, columns=feat_names)
    y_train_series = pd.Series(y_train)
    corr = df.corrwith(y_train_series)
    
    selected_features =  corr_filter(corr,n_features=12, upper_threshold=0.55, lower_threshold=0.001, tighten=0.01, max_iterations=100) #optimized with other script


    print(f"Selected Features:\t{len(selected_features)}\n\t{', '.join(selected_features)}\n")
    selected_indices = [feat_names.index(feature) for feature in selected_features]

    # Should n_features == len(selected_indices) == len(selected_features)?
    return selected_indices

def evaluate(predictions, y_test):
    rmse = compute_rmse(predictions, y_test)    # Lower the better
    print(f"Root Mean Squared Error (RMSE):\n\t{rmse}\n")
    r2 = r2_score(y_test, predictions)          # Higher the better
    print(f"R² Score:\n\t{r2}\n")

    return rmse, r2

def plot(predictions, y_test, selected_feats_idx, model):
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    
    slope, intercept = np.polyfit(y_test, predictions, 1)
    y_line = slope * y_test + intercept
    plt.plot(y_test, y_line, color='red', label='Regression Line')

    plt.show()

def predict(path_to_data, model, selected_feats_idx):
    new_data = pd.read_csv(path_to_data)

    new_data = new_data.drop(columns=['img_filename'])   
    new_data_preprocessed, _ = preprocess_data(new_data)

    
    new_data_selected = new_data_preprocessed[:, selected_feats_idx]
    new_predictions = model.predict(new_data_selected)
    pd.DataFrame(new_predictions).to_csv(r'y_pred.csv', index=False, header=False, float_format='%.10f')
    print("Predictions written!", '\n')


def main():
    # Load the data (adjust paths as needed)
    X_train, X_test, y_train, y_test = load_data()

    ##################
    # preprocesing data
    ###################
    X_train = X_train.drop(columns=['img_filename'])
    X_test = X_test.drop(columns=['img_filename'])


    X_train_preprocessed, feature_names = preprocess_data(X_train)
    X_test_preprocessed, _ = preprocess_data(X_test)
  
    # init LR model
    selected_feats_idx = prepare_model(X_train_preprocessed, y_train, feature_names)
    model = LinearRegression()

    # Filter only the selected features columns
    X_train_selected = X_train_preprocessed[:, selected_feats_idx]
    X_test_selected = X_test_preprocessed[:, selected_feats_idx]

    # Train the model
    model.fit(X_train_selected, y_train)

    # Predict
    predictions = model.predict(X_test_selected)

    evaluate(predictions, y_test)

    #plots
    plot(predictions, y_test, selected_feats_idx, model)

    # Use the model to predict the new data
    predict(r'..\data_students\unlabeled_data\X.csv', model, selected_feats_idx)

if __name__ == "__main__":
    main()