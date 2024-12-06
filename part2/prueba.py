from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from TP_machine_UCL.part2.brainy import preprocess_data, load_data


# Generate synthetic data
X,_, y,_ = load_data()

X = X.drop(columns=['img_filename'])


X_train_preprocessed, feature_names = preprocess_data(X)

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
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
grid_search.fit(X_train_preprocessed, y_train_scaled)

# Output best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
