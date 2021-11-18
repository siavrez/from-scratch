# Importing the required modules
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from typing import Callable
from functools import partial
from sklearn.model_selection import train_test_split as tta


def eucledian_distance(p1: np.array, p2: np.array):
    return np.sqrt(np.sum((p1-p2)**2))


def knn_predict(train: np.array, target: np.array, test: np.array, n_neighbors: int = 7, distance_metric: Callable = eucledian_distance):
    predicted_labels = []
    for item in test:
        point_dist = np.array(
            list(map(partial(distance_metric, p2=item), train)))
        nearest_points_idx = np.argsort(point_dist)[:n_neighbors]
        label = mode(target[nearest_points_idx]).mode[0]
        predicted_labels.append(label)

    return predicted_labels


def knn_predict_v2(train: np.array, target: np.array, test: np.array, n_neighbors: int = 7, distance_metric: callable = eucledian_distance):

    return [mode(target[np.argsort(np.array(list(map(partial(distance_metric, p2=item), train))))[:n_neighbors]]).mode[0] for item in test]

def test_classifier(prdictor:Callable) -> np.ndarray:
    iris = load_iris()
    X_train, X_test, y_train, y_test = tta(
        iris.data, iris.target, test_size=0.3, random_state=0, shuffle=True)
    y_pred = knn_predict(X_train, y_train, X_test, 7)
    return y_test, y_pred

# Checking the accuracy
if __name__=='__main__':
    y_test, y_pred = test_classifier(knn_predict)
    print(accuracy_score(y_test, y_pred))
