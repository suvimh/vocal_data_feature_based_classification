import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt


def prepare_data(csv_file, audio_source, classify, algorithm='mlp', feature_selection=1, modalities=None, num_features=None, conditions_to_remove=None, column_name=None):
    """
    Prepare the data by loading, cleaning, and selecting features.

    Args:
        csv_file (str): The path to the csv file containing the data.
        audio_source (str): The name of the audio source.
        classify (str): The column name of the target variable to classify.
        algorithm (str): The algorithm to use for feature selection. Default is 'mlp'.
        feature_selection (int): The feature selection method to use. Default is 1.
        modalities (list): A list of modalities to consider. Default is None.
        num_features (int): The number of features to select. Default is None.
        conditions_to_remove (list): The conditions to remove from the dataset. Default is None.
        column_name (str): The name of the column to check for conditions to remove. Default is None.

    Returns:
        x (DataFrame): The prepared feature set.
        y (Series): The target variable.
        selected_features (list): The selected features (if feature selection is applied).
    """
    data = pd.read_csv(csv_file)

    metadata_columns = [
        "recording_condition",
        "phrase",
        "clip_number",
        "phonation",
        f"{audio_source}_note",
    ]

    x = data.drop(columns=metadata_columns)

    if modalities:
        x = extract_modality(modalities, x)

    x = handle_missing_data(x)
    y = data[classify]

    if conditions_to_remove:
        if column_name:
            x, y = remove_specified_conditions(x, y, conditions_to_remove, column_name)
            for condition in y:
                if condition in conditions_to_remove:
                    raise ValueError(f"Condition {condition} was not removed")
        else:
            raise ValueError("Column name must be provided when removing conditions")

    selected_features = None
    if num_features:
        if feature_selection == 1:
            print(1)
            x, selected_features = feature_selection_v1(num_features, x, y)
        elif feature_selection == 2:
            x, selected_features = feature_selection_v2(num_features, x, y, algorithm)
        print("Selected features: ", selected_features)

    return x, y


def handle_missing_data(x):
    """
    Handle missing data by filling NaN values with zeros and ensuring consistent handling
    of missing pose landmarks across conditions. Pose landmakrs with missing data need to be
    handled differently to accommodate for possible different camera angles.

    Args:
        x (DataFrame): The input features.

    Returns:
        x (DataFrame): The input features with missing data handled.
    """
    # Identify columns with pose landmarks --
    pose_landmark_columns = [col for col in x.columns if "pose_landmark" in col]
    # Identify columns with NaNs before filling them
    nan_columns = x[pose_landmark_columns].isna().any(axis=0)

    # Replace all NaN values with zeros
    x = x.fillna(0)

    # Ensure consistent handling of missing pose landmarks across conditions
    for col in nan_columns.index:
        if nan_columns[col]:
            parts = col.split("_")
            landmark_number = parts[-2]  # Extract landmark number
            source = parts[0]  # Extract source (e.g., computer)
            columns_to_update = [
                f"{source}_pose_landmark_{landmark_number}_{axis}"
                for axis in ["x", "y", "z"]
            ]
            x[columns_to_update] = 0
    return x


def extract_modality(modalities, x):
    """
    Extract the specified modalities from the feature set.

    Args:
        modalities (list): The modalities to extract.
        x (DataFrame): The input features.

    Returns:
        x (DataFrame): The input features with the specified modalities extracted.
    """
    # Define columns to be dropped for each modality
    audio_columns_to_drop = [
        col for col in x.columns if any(substr in col for substr in ["spec", "mfcc", "tristimulus", "rms", "pitch"])
    ]
    video_columns_to_drop = [col for col in x.columns if "landmark" in col]
    biosignal_columns_to_drop = ["emg_1", "respiration_1"]

    # Check which modalities are not in the list and prepare columns to drop
    if "audio" not in modalities:
        x = x.drop(columns=audio_columns_to_drop)
    if "video" not in modalities:
        x = x.drop(columns=video_columns_to_drop)
    if "biosignals" not in modalities:
        x = x.drop(columns=biosignal_columns_to_drop)
    
    return x


def remove_specified_conditions(x, y, conditions_to_remove, column_name):
    """
    Remove specified conditions from the dataset based on a given column.

    Args:
        x (DataFrame): The input features.
        y (Series): The target variable.
        conditions_to_remove (list): The conditions to remove.
        column_name (str): The name of the column to check for conditions to remove.

    Returns:
        x (DataFrame): The input features with the specified conditions removed.
        y (Series): The target variable with the specified conditions removed.
    """
    # Concatenate x and y to ensure they stay aligned during filtering
    data = pd.concat([x, y], axis=1)

    for condition in conditions_to_remove:
        data = data[data[column_name] != condition]

    x_filtered = data.drop(columns=[column_name])
    y_filtered = data[y.name]

    return x_filtered, y_filtered


def standardize_x_data(X_train, X_test):
    """
    Standardizes the features in the train and test datasets separately.

    Args:
        X_train (DataFrame): Training feature set.
        X_test (DataFrame): Testing feature set.

    Returns:
        X_train_std (DataFrame): Standardized training feature set.
        X_test_std (DataFrame): Standardized testing feature set.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.
    """
    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(X_train)
    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)

    X_test_std = scaler.transform(X_test)
    X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)

    return X_train_std, X_test_std


def feature_selection_v1(num_features, x, y):
    """
    Select the top num_features using the SelectKBest method.

    Args:
        num_features (int): The number of features to select.
        X (DataFrame): The input features.
        y (Series): The target variable.

    Returns:
        X_new (DataFrame): The selected features.
        selected_features (list): The names of the selected features.
    """
    # Drop constant columns before feature selection
    constant_columns = [col for col in x.columns if x[col].nunique() == 1]
    x = x.drop(columns=constant_columns)

    selector = SelectKBest(f_classif, k=num_features)
    x_new = selector.fit_transform(x, y)
    selected_mask = selector.get_support()
    selected_features = x.columns[selected_mask]

    return x_new, selected_features


def feature_selection_v2(max_features, x, y, algorithm, hidden_layers=(12,), random_state=42):
    """
    Select the best performing feature combination up to max_features using SelectKBest and cross-validation.

    Args:
        max_features (int): The maximum number of features to select.
        x (DataFrame): The input features.
        y (Series): The target variable.
        algorithm (str): The algorithm to evaluate ('knn' or 'mlp').
        hidden_layers (tuple): The number of hidden layers and units in each layer for the MLP classifier.
        random_state (int): The random state for reproducibility.

    Returns:
        x_new (DataFrame): The input features with the best selected features.
        selected_features (list): The names of the selected features.
    """
    # Drop constant columns before feature selection
    constant_columns = [col for col in x.columns if x[col].nunique() == 1]
    x = x.drop(columns=constant_columns)
    
    best_score = -float('inf')
    selected_features = None
    
    # Iterate over number of features to select
    for k in range(1, max_features + 1):
        selector = SelectKBest(f_classif, k=k)
        x_new = selector.fit_transform(x, y)
        selected_mask = selector.get_support()
        selected_features = x.columns[selected_mask]

        # Evaluate the performance using cross-validation for the specified algorithm
        if algorithm == 'knn':
            model = KNeighborsClassifier()
        elif algorithm == 'mlp':
            model = MLPClassifier(  hidden_layer_sizes=hidden_layers,
                                    learning_rate_init=0.01,
                                    momentum=0.5,
                                    max_iter=2000,
                                    random_state=random_state)
        else:
            raise ValueError("Unsupported algorithm. Choose 'knn' or 'mlp'.")

        scores = cross_val_score(model, x_new, y, cv=5)
        mean_score = scores.mean()
        
        # Update best score and features if current combination is better
        if mean_score > best_score:
            best_score = mean_score
            selected_features = selected_features
    
    # Select the best features
    x_new = x[selected_features]
    
    return x_new, list(selected_features)


def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plot the confusion matrix as a heatmap.

    Args:
        conf_matrix (array): The confusion matrix to plot.
        class_names (array): The class names.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def calculate_metrics_2_class(conf_matrix):
    """
    Calculate precision, recall, specificity, and false positive rate from the confusion matrix.

    Args:
        conf_matrix (array): The confusion matrix.

    Returns:
        precision (float): The precision of the model.
        recall (float): The recall of the model.
        specificity (float): The specificity of the model.
        fpr (float): The false positive rate of the model.
    """
    TN, FP, FN, TP = conf_matrix.ravel()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    fpr = FP / (FP + TN)

    return precision, recall, specificity, fpr


def calculate_metrics_multi_class(y_true, y_pred):
    """
    Calculate precision and recall from the predictions and true labels.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        conf_matrix (array, optional): Confusion matrix. If provided, specificity and FPR will be calculated from it.

    Returns:
        precision (float): Precision score.
        recall (float): Recall score.
    """
    # Ensure y_true and y_pred are numpy arrays if they are pandas Series
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    return precision, recall
