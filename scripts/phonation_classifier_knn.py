import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_test_knn(csv_file, classify, audio_source, num_features=None, modalities=None, test_size=0.2, n_neighbors=5):
    '''
        Train and test a KNN classifier on the data in the csv file.    

        Args:
            csv_file (str): The path to the csv file containing the data.
            classify (str): The column name of the target variable to classify.
            audio_source (str): The name of the audio source.
            columns_to_drop (list): A list of columns to drop from the data before training the model.
            test_size (float): The proportion of the data to use as the test set. 
                               Default 0.2 for 80-20 train-test split.
            n_neighbors (int): The number of neighbors to use for the KNN classifier.
                               Default 5.

        Returns:
            y_test (array): The true labels for the test set.
            y_pred (array): The predicted labels for the test set.
    '''
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
            columns_to_drop = [col for col in data.columns if any(substr in col for substr in ['spec', 'mfcc', 'tristimulus', 'rms', 'pitch'])]
            x = x.drop(columns=columns_to_drop)
            # remove biosignal columns
            x = x.drop(columns=["emg", "pzt"])
        elif "biosignals" in modalities:
            # remove audio columns
            columns_to_drop = [col for col in data.columns if any(substr in col for substr in ['spec', 'mfcc', 'tristimulus', 'rms', 'pitch'])]
            x = x.drop(columns=columns_to_drop)
            # remove video columns
            x = x.drop(columns=[col for col in x.columns if 'landmark' in col])

    y = data[classify]

    if num_features:
        x, selected_features = feature_selection(num_features, x, y)
        print(selected_features)

    class_names = y.unique()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42,  stratify=y)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    
    y_pred = knn.predict(x_test)
    
    return y_test, y_pred, class_names



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

def evaluate_model(y_test, y_pred):
    '''
        Evaluate the performance of the model using accuracy and confusion matrix.

        Args:
            y_test (array): The true labels for the test set.
            y_pred (array): The predicted labels for the test set.
        
        Returns:
            accuracy (float): The accuracy of the model.
            conf_matrix (array): The confusion matrix of the model.
    '''
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, conf_matrix


def plot_confusion_matrix(conf_matrix, class_names):
    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()