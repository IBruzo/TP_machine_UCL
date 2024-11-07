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
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV



# Transform text columns into useful digital tables
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

    
    # Combine all features
    X_combined = np.hstack([
        X[numeric_features + ordinal_features].values,  
        X_categorical_encoded 
    ])

    # Normalize & scale the combined features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Extract new professions column names
    encoded_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
    feature_names = numeric_features + ordinal_features + list(encoded_feature_names)

    return X_scaled, feature_names


def compute_rmse(predict, target):
    diff = predict - np.squeeze(target)
    return np.sqrt((diff ** 2).sum() / len(target))


def forward_feature_selection(model, X, y, n_features):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, direction='forward', cv=5)
    sfs.fit(X, y)
    selected_indices = np.where(sfs.get_support())[0]
    return selected_indices

def load_data():
    # Load the data (adjust paths as needed)
    X_train = pd.read_csv(r'..\data_students\labeled_data\X_train.csv')
    X_test = pd.read_csv(r'..\data_students\labeled_data\X_test.csv')
    y_train = pd.read_csv(r'..\data_students\labeled_data\y_train.csv', header=None).values.ravel()
    y_test = pd.read_csv(r'..\data_students\labeled_data\y_test.csv', header=None).values.ravel()

    return X_train, X_test, y_train, y_test

def prepare_model(X, y_train, feat_names):

    def datafr_mutinfo_featsel(X, feat_names, y_train):
        df = pd.DataFrame(X, columns=feat_names)
        mi = pd.Series(mutual_info(df, y_train), index=df.columns)

        # TODO: Select n features based on the function call
        n_features = 8
        return df, mi, n_features
    
    X_train_preprocessed_df, mi, n_features = datafr_mutinfo_featsel(X, feat_names, y_train)
    print(f"Mutual Information Scores:\n {mi}\n")

    # Filter out the features with mi = 0
    selected_features = mi[mi > 0.05].index.tolist()       
    # for now we are selecting all features with mi > 0, but could be consider for optimization purposes a threshold of ~>0.03

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

    # Define parameter grid
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (150, 100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.005, 0.01],
        'max_iter': [500],
        'solver': ['adam'],
        'batch_size': [16, 32, 64]
    }

    # Initialize MLPRegressor and GridSearchCV
    model = MLPRegressor(random_state=0)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_preprocessed, y_train)

    # Output best parameters and score
    print("Best parameters:", grid_search.best_params_)
    print("Best RMSE:", (-grid_search.best_score_) ** 0.5)

    model = MLPRegressor(random_state=0, **grid_search.best_params_)
    
    # Filter only the selected features columns
    X_train_selected = X_train_preprocessed[:, selected_feats_idx]
    X_test_selected = X_test_preprocessed[:, selected_feats_idx]
    

    # Train the model
    model.fit(X_train_selected, y_train)

    # Predict
    predictions = model.predict(X_test_selected)

    evaluate(predictions, y_test)

    #plots
    #plot(predictions, y_test, selected_feats_idx, model)

    # Use the model to predict the new data
    predict(r'..\data_students\unlabeled_data\X.csv', model, selected_feats_idx)

if __name__ == "__main__":
    main()