# Importing the required modules
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing
from typing import Callable, Dict, Tuple
from functools import partial
from sklearn.model_selection import train_test_split as tta
from sklearn.preprocessing import StandardScaler


def eucledian_distance(p1: np.array, p2: np.array):
    """[summary]

    Args:
        p1 (np.array): [description]
        p2 (np.array): [description]

    Returns:
        [type]: [description]
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))


def knn_predict(
    train: np.array,
    target: np.array,
    test: np.array,
    n_neighbors: int = 7,
    distance_metric: Callable = eucledian_distance,
):
    """[summary]

    Args:
        train (np.array): [description]
        target (np.array): [description]
        test (np.array): [description]
        n_neighbors (int, optional): [description]. Defaults to 7.
        distance_metric (Callable, optional): [description]. Defaults to eucledian_distance.

    Returns:
        [type]: [description]
    """
    predicted_labels = []
    for item in test:
        point_dist = np.array(list(map(partial(distance_metric, p2=item), train)))
        nearest_points_idx = np.argsort(point_dist)[:n_neighbors]
        label = mode(target[nearest_points_idx]).mode[0]
        predicted_labels.append(label)

    return predicted_labels


def knn_predict_v2(
    train: np.array,
    target: np.array,
    test: np.array,
    n_neighbors: int = 7,
    distance_metric: callable = eucledian_distance,
):
    """[summary]

    Args:
        train (np.array): [description]
        target (np.array): [description]
        test (np.array): [description]
        n_neighbors (int, optional): [description]. Defaults to 7.
        distance_metric (callable, optional): [description]. Defaults to eucledian_distance.

    Returns:
        [type]: [description]
    """
    return [
        mode(
            target[
                np.argsort(
                    np.array(list(map(partial(distance_metric, p2=item), train)))
                )[:n_neighbors]
            ]
        ).mode[0]
        for item in test
    ]


def lr_coefficients(train: np.array, target: np.array) -> Tuple[int, np.array]:
    """[summary]

    Args:
        train (np.array): [description]
        target (np.array): [description]

    Returns:
        Tuple[int, np.array]: [description]
    """
    x_mean, y_mean = np.mean(train), np.mean(target)
    b1 = sum(np.cov(train[:, i], target)[0, 0] for i in range(train.shape[1])) / np.var(
        train
    )
    b0 = y_mean - b1 * x_mean
    return b0, b1


def simple_linear_regression_predict(train: np.array, target: np.array, test: np.array):
    """[summary]

    Args:
        train (np.array): [description]
        target (np.array): [description]
        test (np.array): [description]
    """
    predictions = []
    b0, b1 = lr_coefficients(train, target)
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
    return predictions


def test_predictor(
    predictor: Callable, classification: bool = True, kwargs: Dict = {}
) -> np.ndarray:
    """[summary]

    Args:
        predictor (Callable): [description]
        classification (bool, optional): [description]. Defaults to True.
        kwargs (Dict, optional): [description]. Defaults to {}.

    Returns:
        np.ndarray: [description]
    """
    if classification:
        dataset = load_iris()
    else:
        dataset = fetch_california_housing()
    X_train, X_test, y_train, y_test = tta(
        dataset.data, dataset.target, test_size=0.3, random_state=0, shuffle=True
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_pred = predictor(X_train, y_train, X_test, **kwargs)
    return y_test, y_pred


# Checking the accuracy
if __name__ == "__main__":
    y_test, y_pred = test_predictor(knn_predict, {"n_neighbors": 7})
    print(accuracy_score(y_test, y_pred))
    y_test, y_pred = test_predictor(
        simple_linear_regression_predict, classification=False
    )
    print(mean_squared_error(y_test, y_pred))
