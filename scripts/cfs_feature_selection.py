import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

def cfs_feature_selection(x, y, num_features):
    """
    Perform Correlation-based Feature Selection (CFS).

    Args:
        x (DataFrame): The feature matrix.
        y (Series): The target variable.
        num_features (int): The number of features to select.

    Returns:
        x_selected (DataFrame): Selected feature matrix.
        selected_features (list): List of selected feature names.
    """
    # Convert y to numeric labels if necessary
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    # Calculate feature-class correlations
    feature_class_corr = []
    for feature in x.columns:
        corr = np.abs(np.corrcoef(x[feature], y)[0, 1])  # Absolute correlation with class
        feature_class_corr.append((feature, corr))

    # Sort features by correlation with class label (descending)
    feature_class_corr.sort(key=lambda x: x[1], reverse=True)

    # Initialize selected features list and feature set
    selected_features = []
    feature_set = set()

    # Add the first feature (highest correlation with class)
    selected_features.append(feature_class_corr[0][0])
    feature_set.add(feature_class_corr[0][0])

    # Iterate until desired number of features is selected
    while len(selected_features) < num_features:
        max_cfs = -float('inf')
        best_feature = None

        # Evaluate CFS criterion for each remaining feature
        for feature, _ in feature_class_corr:
            if feature not in feature_set:
                new_feature_set = feature_set.union([feature])
                cfs_value = calculate_cfs(x[list(new_feature_set)], y)

                # Update best feature based on CFS value
                if cfs_value > max_cfs:
                    max_cfs = cfs_value
                    best_feature = feature

        # Add the best feature to the selected features list and set
        selected_features.append(best_feature)
        feature_set.add(best_feature)

    # Return selected features and corresponding feature matrix
    x_selected = x[selected_features]
    return x_selected, selected_features

def calculate_cfs(x_subset, y):
    """
    Calculate the CFS value for a subset of features.

    Args:
        x_subset (DataFrame): Subset of feature matrix.
        y (Series): The target variable.

    Returns:
        cfs_value (float): CFS criterion value.
    """
    # Calculate feature-feature correlations
    correlations = x_subset.corr().abs()

    # Calculate feature-class correlations
    f_class = np.abs(np.corrcoef(x_subset.T, y)[0, 1:])

    # Calculate CFS criterion
    num_features = len(x_subset.columns)
    cfs_value = (np.mean(f_class) / ((1 / num_features) * np.sum(np.sum(correlations)))) * num_features

    return cfs_value


# # Perform CFS feature selection
# x_selected, selected_features = cfs_feature_selection(x, y, num_features)
