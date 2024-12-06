import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression as mutual_info
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
import seaborn as sns


def preprocess_data(X):
    # Create BMI from height and weight
    X['BMI'] = X['weight'] / ((X['height'] / 100) ** 2)  
    
    # Remove height and weight columns
    X = X.drop(columns=['height', 'weight'])
    
    # Updated list of numeric features
    numeric_features = ['age', 'blood pressure', 'calcium', 'cholesterol', 'hemoglobin', 'potassium', 'vitamin D', 'BMI']
    
    # Map ordinal features
    ordinal_features = ['sarsaparilla', 'smurfberry liquor', 'smurfin donuts']
    feat_score = {"Very low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very high": 5}
    for feature in ordinal_features:
        X[feature] = X[feature].map(feat_score)
    
    # One-hot encode categorical features
    categorical_features = ['profession']
    onehot_encoder = OneHotEncoder(sparse_output=False)
    X_categorical_encoded = onehot_encoder.fit_transform(X[categorical_features])
    
    # Normalize & scale the combined features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numeric_features + ordinal_features].values)
    
    # Combine scaled numeric/ordinal features with one-hot encoded categorical features
    X_combined = np.hstack([X_scaled, X_categorical_encoded])
    
    # Extract new feature names
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
    selected_features = mi[mi > 0.0].index.tolist()       
    # for now we are selecting all features with mi > 0, but could be consider for optimization purposes a threshold of ~>0.03

    print(f"Selected Features:\t{len(selected_features)}\n\t{', '.join(selected_features)}\n")
    selected_indices = [feat_names.index(feature) for feature in selected_features]


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

def exploratory_analysis(data, targets, feature_names):
   
    
    targets = pd.Series(targets.ravel(), name="risk")  # Flatten 
    data_df = pd.DataFrame(data, columns=feature_names)

    # Check if the lengths match
    if len(data_df) != len(targets):
        raise ValueError(f"Length mismatch: data has {len(data_df)} rows, but targets has {len(targets)} entries")

    
    data_df['risk'] = targets

    # Abbreviate feature names for readability
    abbreviated_names = {
        'age': 'Age', 'blood pressure': 'Blood pressure', 'calcium': 'Ca', 'cholesterol': 'Cholesterol', 
        'hemoglobin': 'hemoglobin', 'potassium': 'K', 'vitamin D': 'Vitamin D', 'BMI': 'BMI',
        'sarsaparilla': 'sarsaparilla', 'smurfberry liquor': 'SmurfLiquor', 'smurfin donuts': 'SmurfDonuts'
    }
    
    # Abbreviate profession one-hot encoded feature names
    for col in data_df.columns:
        if col.startswith('profession_'):
            abbreviated_names[col] = col.replace('profession_', 'Prof_')
    
    data_df = data_df.rename(columns=abbreviated_names)
    
    # Correlation heatmap
    plt.figure(figsize=(14, 12))  
    correlation_matrix = data_df.corr()  
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                annot_kws={"size": 10}, 
                cbar_kws={'label': 'Correlation'},  
                xticklabels=correlation_matrix.columns,  
                yticklabels=correlation_matrix.columns) 

    # Rotate x-axis and y-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks( va="center")
    plt.title("Feature Correlation with Heart Failure Risk", fontsize=16)
    plt.tight_layout(pad=2.0)  # Adjust layout to prevent label overlap
    plt.show()

    # Distribution of cholesterol by risk levels
    sns.boxplot(x='risk', y='Cholesterol', data=data_df)
    plt.title("Cholesterol Levels by Risk (Log Scale)")
    plt.xticks(rotation=45, ha="right")  
    plt.show()

    # Feature importance visualization
    feat_score = mutual_info(data_df.drop(columns=['risk']), targets)
    plt.barh(data_df.columns.drop('risk'), feat_score, color='blue')
    plt.xlabel("Mutual Information Score")
    plt.title("Feature Importance")
    plt.show()


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

    # Perform EDA
    exploratory_analysis(X_train_preprocessed, y_train,feature_names)
  
    # init LR model
    selected_feats_idx = prepare_model(X_train_preprocessed, y_train, feature_names)

    model = MLPRegressor(hidden_layer_sizes=(100,50 ), max_iter=500, random_state=42, learning_rate_init=0.01, alpha=0.1, solver='adam', batch_size=16, activation='tanh')

    # Filter only the selected features columns
    X_train_selected = X_train_preprocessed[:, selected_feats_idx]
    X_test_selected = X_test_preprocessed[:, selected_feats_idx]
    # Scale the target
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    

    # Train the model
    model.fit(X_train_selected, y_train_scaled)

    # Predict
    predictions_scaled = model.predict(X_test_selected)

    # Unscale the predictions
    predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
    evaluate(predictions, y_test)

    #plots
    plot(predictions, y_test, selected_feats_idx, model)

    # Use the model to predict the new data
    predict(r'..\data_students\unlabeled_data\X.csv', model, selected_feats_idx)

if __name__ == "__main__":
    main()