import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,  f1_score
from scripts.utils import prepare_data


def train_and_test_knn(csv_file, classify, audio_source, num_features=None, modalities=None, test_size=0.2, n_neighbors=5):
    '''
    Train and test a KNN classifier on the data in the csv file.    

    Args:
        csv_file (str): The path to the csv file containing the data.
        classify (str): The column name of the target variable to classify.
        audio_source (str): The name of the audio source.
        test_size (float): The proportion of the data to use as the test set. Default is 0.2 for 80-20 train-test split.
        n_neighbors (int): The number of neighbors to use for the KNN classifier. Default is 5.

    Returns:
        y_test (array): The true labels for the test set.
        y_pred (array): The predicted labels for the test set.
    '''
    x, y = prepare_data(csv_file, audio_source, classify, algorithm='knn', modalities=modalities, num_features=num_features)

    class_names = y.unique()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    
    y_pred = knn.predict(x_test)
    
    return y_test, y_pred, class_names


def evaluate_model(y_test, y_pred):
    '''
        Evaluate the performance of the model using accuracy, F1-score, and confusion matrix.

        Args:
            y_test (array): The true labels for the test set.
            y_pred (array): The predicted labels for the test set.
        
        Returns:
            accuracy (float): The accuracy of the model.
            f1 (float): The F1-score of the model.
            conf_matrix (array): The confusion matrix of the model.
    '''
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Calculate F1-score
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, f1, conf_matrix