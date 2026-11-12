import numpy as np
import torch

from eicore.common import EPS


def minmax_normalize(x, eps=EPS):
    x = np.asarray(x, dtype=np.float64)
    mins = x.min(axis=0)
    spans = x.max(axis=0) - mins
    safe_spans = np.where(spans > eps, spans, 1.0)
    return (x - mins) / safe_spans, mins, safe_spans


def one_hot(labels, n_classes):
    labels = np.asarray(labels, dtype=np.int64)
    out = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def load_iris_tensors(seed):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    x_raw = np.asarray(iris.data, dtype=np.float64)
    feature_mins = x_raw.min(axis=0)
    feature_widths = x_raw.max(axis=0) - feature_mins
    y_onehot = one_hot(iris.target, 3)
    x_train, x_test, y_train, y_test = train_test_split(
        x_raw,
        y_onehot,
        test_size=0.2,
        random_state=seed,
        stratify=iris.target,
    )
    return (
        torch.tensor(x_train, dtype=torch.float64),
        torch.tensor(y_train, dtype=torch.float64),
        torch.tensor(x_test, dtype=torch.float64),
        torch.tensor(y_test, dtype=torch.float64),
        torch.tensor(feature_mins, dtype=torch.float64),
        torch.tensor(feature_widths, dtype=torch.float64),
    )
