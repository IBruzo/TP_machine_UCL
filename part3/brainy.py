import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import mutual_info_regression as mutual_info
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from TP_machine_UCL.part3.clases import MyCNN, CustomDataset, SimpleCNN
from TP_machine_UCL.part3.utils import visualize_dataset_tSNE



# Transform text columns into useful digital tables
def preprocess_data(X,n_features_img):

    # Table for features
    numeric_features = ['age', 'blood pressure', 'calcium', 'cholesterol', 'hemoglobin', 'height', 'potassium', 'vitamin D', 'weight','sarsaparilla', 'smurfberry liquor', 'smurfin donuts']
    # Add image features to numeric features
    for i in range(n_features_img):
        numeric_features.append(f'img_feat_{i+1}')
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

    def datafr_mutinfo_featsel(X, feat_names, y_train):
        df = pd.DataFrame(X, columns=feat_names)
        mi = pd.Series(mutual_info(df, y_train), index=df.columns)

        # TODO: Select n features based on the function call
        n_features = 8
        return df, mi, n_features
    
    X_train_preprocessed_df, mi, n_features = datafr_mutinfo_featsel(X, feat_names, y_train)
    print(f"Mutual Information Scores:\n {mi}\n")

    # Filter out the features with mi = 0
    selected_features = mi[mi > 0.03].index.tolist()       
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

def predict(path_to_data, model, selected_feats_idx,cnn,n_features_img):
    new_data = pd.read_csv(path_to_data)

    new_data_imgs = new_data["img_filename"].values   

    dataset = CustomDataset(new_data_imgs, r'..\data_students\unlabeled_data\Img')

    img_features_train = cnn.extract_features(dataset.images, dataset.images_directory)
    img_features_train_df = pd.DataFrame(
    img_features_train, 
    columns=[f'img_feat_{i+1}' for i in range(img_features_train.shape[1])]
    )
    new_data = pd.concat([new_data, img_features_train_df], axis=1)


    new_data_preprocessed, _ = preprocess_data(new_data,n_features_img)

    
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
    #X_train = X_train.drop(columns=['img_filename'])
    images_test = X_test["img_filename"].values
    images_train = X_train["img_filename"].values
    # Define the number of features to extract 
    n_features_img=8
    # Create instance of cnn model
    #n_features=n_features_img, batch_size=50, n_epochs=20, learning_rate=0.0005
    cnn = MyCNN(n_features=n_features_img, batch_size=32, n_epochs=20, learning_rate=0.0005) 

    # Fit cnn
    cnn.fit(images_train, y_train, r'..\data_students\labeled_data\Img_train')

    dataset_train = CustomDataset(images_train, r'..\data_students\labeled_data\Img_train', target=y_train)

    img_features_train = cnn.extract_features(dataset_train.images, dataset_train.images_directory)
    img_features_train_df = pd.DataFrame(
    img_features_train, 
    columns=[f'img_feat_{i+1}' for i in range(img_features_train.shape[1])]
    )
    X_train = pd.concat([X_train, img_features_train_df], axis=1)



    dataset_test = CustomDataset(images_test, r'..\data_students\labeled_data\Img_test', target=y_test)

    img_features_test = cnn.extract_features(dataset_test.images, dataset_test.images_directory)
    img_features_test_df = pd.DataFrame(
    img_features_test, 
    columns=[f'img_feat_{i+1}' for i in range(img_features_test.shape[1])]
    )

    X_test = pd.concat([X_test, img_features_test_df], axis=1)

    X_train_preprocessed, feature_names = preprocess_data(X_train,n_features_img)
    X_test_preprocessed, _ = preprocess_data(X_test,n_features_img)
  
    # init LR model
   # selected_feats_idx = prepare_model(X_train_preprocessed, y_train, feature_names)

    #Best Parameters: {'activation': 'relu', 'alpha': 0.01, 'batch_size': 16, 'hidden_layer_sizes': (100, 50), 'learning_rate_init': 0.001, 'solver': 'adam'}
    #Best Score: -0.0020914436665120115
    #TODO x alguna razon es peor?
    #hidden_layer_sizes=(100, ), max_iter=500, random_state=42, learning_rate_init=0.001, alpha=0.1, solver='adam', batch_size=32, activation='relu'
    #MAS HIDDEN LAYES ES PEOR >:((
    model = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=500, random_state=42, learning_rate_init=0.01, alpha=0.1, solver='adam', batch_size=16, activation='relu')

    # Filter only the selected features columns
   # X_train_selected = X_train_preprocessed[:, selected_feats_idx]
   #X_test_selected = X_test_preprocessed[:, selected_feats_idx]

   # Scale the target
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()



    #plots
   # plot(predictions, y_test, selected_feats_idx, model)
    best_score = float('+inf')
    best_predictions = None
    scores = []

    for i in range(10):
        selected_feats_idx = prepare_model(X_train_preprocessed, y_train, feature_names)
        X_train_selected = X_train_preprocessed[:, selected_feats_idx]
        X_test_selected = X_test_preprocessed[:, selected_feats_idx]
        print(f"Iteration {i+1}")
        model.fit(X_train_selected, y_train_scaled)
        predictions_scaled = model.predict(X_test_selected)
        predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        rmse, r2 = evaluate(predictions, y_test)
        scores.append(rmse)
        if rmse < best_score:
            best_score = rmse
            best_predictions = predictions
            predict(r'..\data_students\unlabeled_data\X.csv', model, selected_feats_idx,cnn,n_features_img)

    average_score = np.mean(scores)
    print(f"Best rmse Score: {best_score}")
    print(f"Average rmse Score: {average_score}")


    visualize_dataset_tSNE(dataset_train, extract_features=True, feature_extractor=cnn)
    

    # Plot the best predictions
    plot(best_predictions, y_test, selected_feats_idx, model)

    # Use the model to predict the new data
    

if __name__ == "__main__":
    main()
