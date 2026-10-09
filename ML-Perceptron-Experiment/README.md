# Perceptron Reconstructed

This module contains the modern machine learning pipeline for the Perceptron Reconstructed project. It is an advanced implementation of linear classification utilizing the Scikit-Learn framework.

## Pipeline Components

* **Data Augmentation**: Injects Gaussian noise into the training samples to enhance the generalization capacity of the model and prevent overfitting.
* **Automatic Feature Selection**: Utilizes SelectKBest to analyze and select the most informative statistical features from the dataset.
* **Dimensionality Reduction**: Implements Principal Component Analysis (PCA) to compress high dimensional data structures into their most significant orthogonal components.
* **Kernel Transformations**: Applies Radial Basis Function (RBF) approximations, allowing the linear perceptron to classify non-linear relationships.
* **Hyperparameter Optimization**: Conducts exhaustive cross-validation through GridSearchCV to isolate optimal regularization and penalty parameters.

## Execution

Ensure that all dependencies (scikit-learn, numpy, pandas, matplotlib) are installed. The pipeline can be executed via the central FastAPI application or run as a standalone script:

```bash
python ML_Perceptron_Implementation.py
```

Running the script directly will initialize a synthetic classification dataset, execute all feature engineering processes, train the optimal model, and output the final evaluation metrics.
