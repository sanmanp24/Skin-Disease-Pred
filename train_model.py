import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import uniform
from tqdm import tqdm
import numpy as np

# Load features and labels
with open('extracted_features.pkl', 'rb') as f:
    features, labels = pickle.load(f)

# Use MiniBatchKMeans to cluster the features
n_clusters = 10  # Set this to the number of distinct classes
batch_size = 100  # Size of each mini-batch
kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)

# Fit the model to the features and transform them
features = kmeans.fit_transform(features)

# Split dataset and scale features
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define SVC classifier
svc = SVC(probability=True, random_state=42)

# Parameter distribution for hyperparameter tuning
param_dist = {
    'C': uniform(0.1, 10),  # Regularization parameter
    'gamma': uniform(0.001, 0.1),  # Kernel coefficient
}

# RandomizedSearchCV with progress bar
n_iter = 10  # Number of iterations for hyperparameter tuning
cv_folds = 3  # Number of cross-validation folds

# Initialize progress bar
with tqdm(total=n_iter, desc="RandomizedSearchCV Progress", unit="iter") as pbar:
    random_search = RandomizedSearchCV(
        svc,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=StratifiedKFold(cv_folds),
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbose=1  # Provide minimal output
    )

    # Fit the model and track progress
    random_search.fit(X_train, y_train)

    # Update progress bar after fitting
    pbar.update(n_iter)

# Best parameters and evaluation
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)

# Test set evaluation
y_pred = random_search.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(random_search.best_estimator_, f)
