from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from TP_machine_UCL.part3.brainy import preprocess_data, load_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from TP_machine_UCL.part3.clases import MyCNN, CustomDataset, SimpleCNN


# Generate synthetic data
X,_, y,_ = load_data()

images_train = X["img_filename"].values

# Define the number of features to extract 
n_features_img=8
# Create instance of cnn model
cnn = MyCNN(n_features=n_features_img, batch_size=50, n_epochs=20, learning_rate=0.0005) 

# Fit cnn
cnn.fit(images_train, y, r'..\data_students\labeled_data\Img_train')

dataset = CustomDataset(images_train, r'..\data_students\labeled_data\Img_train', target=y)

img_features_train = cnn.extract_features(dataset.images, dataset.images_directory)
img_features_train_df = pd.DataFrame(
img_features_train, 
columns=[f'img_feat_{i+1}' for i in range(img_features_train.shape[1])]
)
X = pd.concat([X, img_features_train_df], axis=1)


X_train_preprocessed, feature_names = preprocess_data(X,n_features_img)

# Define the model-
mlp = MLPRegressor(max_iter=500, random_state=42)

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(75,25), (100,), (100,50)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01],
    'alpha': [ 0.01, 0.1],
    'solver': ['adam'],
    'batch_size': [16,32]
}

kf= KFold(n_splits= 5)


# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_preprocessed, y)

# Output best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
