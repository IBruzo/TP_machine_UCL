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
  
    #inti model

    model = LinearRegression()

    ##################
    # feature selection  mutual info and forward search
    ###################
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)


    mi = pd.Series(mutual_info(X_train_preprocessed_df, y_train), index=X_train_preprocessed_df.columns)
    print(mi)

    n_features = 8

    def mi_filter(mi,n_features):
        mi_copy = mi.copy()
        sorted_mi = mi_copy.abs().sort_values(ascending=False)
        for i in range(len(sorted_mi)):
            if sorted_mi[i] == 0: #if there are values that are 0, we can stop there
                n_features = i
                break
        selected_features = sorted_mi.index[:n_features].tolist()
        return selected_features
    
    
    selected_features =  mi_filter(mi,n_features) 


    print("Selected Features:", selected_features)
    selected_indices = [feature_names.index(feature) for feature in selected_features]

    X_train_selected = X_train_preprocessed[:, selected_indices]
    X_test_selected = X_test_preprocessed[:, selected_indices]

 
    model.fit(X_train_selected, y_train)

    predictions = model.predict(X_test_selected)

    # Evaluation
    rmse = compute_rmse(predictions, y_test)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    r2 = r2_score(y_test, predictions)
    print(f"R² Score: {r2}")

    #plots
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    
    slope, intercept = np.polyfit(y_test, predictions, 1)
    y_line = slope * y_test + intercept
    plt.plot(y_test, y_line, color='red', label='Regression Line')

    plt.show()

    # Predict
    new_data = pd.read_csv(r'..\data_students\unlabeled_data\X.csv')

    new_data = new_data.drop(columns=['img_filename'])   
    new_data_preprocessed, _ = preprocess_data(new_data)

    
    new_data_selected = new_data_preprocessed[:, selected_indices]
    new_predictions = model.predict(new_data_selected)
    pd.DataFrame(new_predictions).to_csv(r'y_pred.csv', index=False, header=False, float_format='%.10f')
    print("Predictions written!")

if __name__ == "__main__":
    main()