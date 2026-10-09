import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.feature_selection import SelectKBest, f_classif

def run_ml_pipeline(config, X_ext=None, y_ext=None):
    if X_ext is not None and y_ext is not None:
        X = X_ext
        y = y_ext
    else:
        # Generating a synthetic dataset
        X, y = make_classification(
            n_samples=config.get("samples", 1000), 
            n_features=config.get("features", 20), 
            n_classes=2, 
            random_state=config.get("seed", 42)
        )
    
    # Data preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Feature Engineering (Interaction feature)
    if X.shape[1] >= 5:
        interaction_feature = X[:, 1] * X[:, 4]
        X = np.c_[X, interaction_feature]
    
    # Automatic Feature Selection
    k_features = config.get("select_k", 5)
    if k_features > X.shape[1]:
        k_features = X.shape[1]
        
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_selected = selector.fit_transform(X, y)
    
    # Data Augmentation by Adding Noise
    noise_level = config.get("noise_level", 0.01)
    if noise_level > 0:
        noisy_X = X_selected + noise_level * np.random.normal(size=X_selected.shape)
        X_augmented = np.concatenate((X_selected, noisy_X), axis=1)
    else:
        X_augmented = X_selected
        
    # Dimensionality Reduction (PCA)
    pca_components = config.get("pca_components", 3)
    pca_enabled = config.get("use_pca", False)
    if pca_enabled and pca_components <= X_augmented.shape[1]:
        pca = PCA(n_components=pca_components)
        X_augmented = pca.fit_transform(X_augmented)
        
    # RBF Kernel Approximation
    rbf_enabled = config.get("use_rbf", False)
    if rbf_enabled:
        rbf_feature = RBFSampler(gamma=1, n_components=config.get("rbf_components", 50), random_state=42)
        X_augmented = rbf_feature.fit_transform(X_augmented)
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_augmented, y, test_size=0.2, random_state=42)
    
    # Model Selection
    model_type = config.get("model_type", "perceptron")
    
    if model_type == "perceptron":
        parameters = {'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01, 0.1]}
        base_model = Perceptron(max_iter=100, eta0=0.1, random_state=42)
        grid_search = GridSearchCV(base_model, parameters, cv=5)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        best_params = {"n_estimators": 50}
    else:
        model = Perceptron(random_state=42)
        model.fit(X_train, y_train)
        best_params = {}
        
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion": confusion_matrix(y_test, y_pred).tolist(),
        "final_features": X_augmented.shape[1],
        "best_params": best_params
    }
    
    return metrics
