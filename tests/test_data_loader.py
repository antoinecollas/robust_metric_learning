import numpy as np
from robust_metric_learning.data_loader import load_data


def _test_dataset(name, N, p, K):
    X, y = load_data(name)
    assert type(X) is np.ndarray
    assert type(y) is np.ndarray
    assert X.shape == (N, p)  # N*p
    assert y.shape == (N,)
    assert (np.unique(y) == np.arange(K)).all()


def test_load_wine():
    _test_dataset('wine', N=178, p=13, K=3)


def test_load_pima():
    _test_dataset('pima', N=768, p=8, K=2)


def test_load_vehicle():
    _test_dataset('vehicle', N=846, p=18, K=4)


def test_load_german():
    _test_dataset('german', N=1000, p=24, K=2)


def test_load_iris():
    _test_dataset('iris', N=150, p=4, K=3)


def test_load_australian():
    _test_dataset('australian', N=690, p=14, K=2)


def test_load_breast_cancer():
    _test_dataset('breast-cancer', N=699, p=9, K=2)


def test_load_segment():
    _test_dataset('segment', N=2310, p=19, K=7)


def test_load_mnist():
    _test_dataset('mnist', N=4000, p=784, K=10)


def test_load_isolet():
    _test_dataset('isolet', N=7797, p=617, K=26)


def test_load_letters():
    _test_dataset('letters', N=20000, p=16, K=26)
