import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def train_mlp(x_train, y_train, hidden_layers=(12,), random_state=42):
    """
    Train an MLP classifier on the data in the csv file after preparing the data.

    Args:
        x_train (DataFrame): The training feature set.
        y_train (Series): The training target variable.
        hidden_layers (tuple): The number of hidden layers and units in each layer. Default is (12,).
        random_state (int): The random state for reproducibility. Default is 42.

    Returns:
        pipeline (Pipeline): The trained pipeline including scaler, feature selection, and MLP classifier.
    """
    # MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers,
                        learning_rate_init=0.01,
                        momentum=0.5,
                        max_iter=2000,
                        random_state=random_state)

    # Pipeline with feature selection and classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', mlp)
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=10)

    # Perform cross-validation on the training set
    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring='f1_weighted')
    #print(f"Cross-validated F1 scores: {cv_scores}")
    print(f"Mean cross-validated F1 score: {cv_scores.mean()}")

    pipeline.fit(x_train, y_train)

    return pipeline


def evaluate_mlp(pipeline, x_test, y_test, class_names):
    '''
        Evaluate the performance of the MLP classifier on the test set.

        Args:
            pipeline (Pipeline): The trained pipeline including scaler, feature selection, and MLP classifier.
            X_test (DataFrame): The test feature set.
            y_test (Series): The true labels for the test set.

        Returns:
            f1 (float): The F1 score of the model on the test set.
            conf_matrix (array): The confusion matrix of the model.
    '''
    y_pred = pipeline.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    class_report = classification_report(y_test, y_pred, target_names=class_names)

    return f1, class_report, conf_matrix