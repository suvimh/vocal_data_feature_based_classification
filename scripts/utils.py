import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt


def prepare_data(csv_file, audio_source, classify, modalities=None, num_features=None):
    """
    Prepare the data by loading, cleaning, and selecting features.

    Args:
        csv_file (str): The path to the csv file containing the data.
        audio_source (str): The name of the audio source.
        classify (str): The column name of the target variable to classify.
        modalities (list): A list of modalities to consider. Default is None.
        num_features (int): The number of features to select. Default is None.

    Returns:
        x (DataFrame): The prepared feature set.
        y (Series): The target variable.
        selected_features (list): The selected features (if feature selection is applied).
    """
    data = pd.read_csv(csv_file)

    metadata_columns = [
        'recording condition',
        'phrase',
        'clip number',
        'frame',
        'phonation',
        f'{audio_source} note'
    ]
    
    x = data.drop(columns=metadata_columns)

    if modalities:
        if "audio" in modalities:
            # remove video columns
            columns_to_drop = [col for col in x.columns if 'landmark' in col]
            x = x.drop(columns=columns_to_drop)
            # remove biosignal columns
            x = x.drop(columns=["emg", "pzt"])
        elif "video" in modalities:
            # remove audio columns
            columns_to_drop = [col for col in x.columns if any(substr in col for substr in ['spec', 'mfcc', 'tristimulus', 'rms', 'pitch'])]
            x = x.drop(columns=columns_to_drop)
            # remove biosignal columns
            x = x.drop(columns=["emg", "pzt"])
        elif "biosignals" in modalities:
            # remove audio columns
            columns_to_drop = [col for col in x.columns if any(substr in col for substr in ['spec', 'mfcc', 'tristimulus', 'rms', 'pitch'])]
            x = x.drop(columns=columns_to_drop)
            # remove video columns
            x = x.drop(columns=[col for col in x.columns if 'landmark' in col])

    y = data[classify]

    selected_features = None
    if num_features:
        x, selected_features = feature_selection(num_features, x, y)
        print("Selected features: ", selected_features)

    return x, y


def standardize_x_data(X_train, X_test):
    '''
    Standardizes the features in the train and test datasets separately.

    Args:
        X_train (DataFrame): Training feature set.
        X_test (DataFrame): Testing feature set.

    Returns:
        X_train_std (DataFrame): Standardized training feature set.
        X_test_std (DataFrame): Standardized testing feature set.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.
    '''
    scaler = StandardScaler()
    
    X_train_std = scaler.fit_transform(X_train)
    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)

    X_test_std = scaler.transform(X_test)
    X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)
    
    return X_train_std, X_test_std


def feature_selection(num_features, X, y):
    '''
        Select the top num_features using the SelectKBest method.

        Args:
            num_features (int): The number of features to select.
            X (DataFrame): The input features.
            y (Series): The target variable.
        
        Returns:
            X_new (DataFrame): The selected features.
            selected_features (list): The names of the selected features.
    '''
    selector = SelectKBest(f_classif, k=num_features)
    x_new = selector.fit_transform(X, y)
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    
    return x_new, selected_features


def plot_confusion_matrix(conf_matrix, class_names):
    '''
        Plot the confusion matrix as a heatmap. 

        Args:
            conf_matrix (array): The confusion matrix to plot.
            class_names (array): The class names.
    '''
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()