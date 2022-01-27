import autograd.numpy as np
import scipy.io as sio
from sklearn.datasets import load_iris, load_wine


def load_data(str_dataset):
    if str_dataset == 'wine':
        data = load_wine()
        X = data.data
        y = data.target
        assert X.shape == (178, 13)  # N*p
        assert y.shape == (178,)
        assert len(np.unique(y)) == 3
    elif str_dataset == 'pima':
        path = 'data/pima.csv'
        data = np.genfromtxt(path, delimiter=' ')
        assert data.shape == (768, 9)
        X = data[:, :8]
        y = data[:, 8]
        assert X.shape == (768, 8)  # N*p
        assert y.shape == (768,)
        assert len(np.unique(y)) == 2
    elif str_dataset == 'vehicle':
        path = 'data/vehicle.csv'
        data = np.genfromtxt(path, delimiter=' ')
        assert data.shape == (846, 19)
        X = data[:, :18]
        y = data[:, 18]
        assert X.shape == (846, 18)  # N*p
        assert y.shape == (846,)
        assert len(np.unique(y)) == 4
    elif str_dataset == 'german':
        path = 'data/german.csv'
        data = np.genfromtxt(path, delimiter=',')
        assert data.shape == (1000, 25)
        X = data[:, :24]
        y = data[:, 24]
        assert X.shape == (1000, 24)  # N*p
        assert y.shape == (1000,)
        assert len(np.unique(y)) == 2
    elif str_dataset == 'australian':
        path = 'data/australian.mat'
        matstruct_contents = sio.loadmat(path)
        X = matstruct_contents['X']
        y = matstruct_contents['y']
        y = y.reshape(-1)
        assert X.shape == (690, 14)  # N*p
        assert y.shape == (690,)
        assert len(np.unique(y)) == 2
    elif str_dataset == 'iris':
        data = load_iris()
        X = data.data
        y = data.target
        assert X.shape == (150, 4)  # N*p
        assert y.shape == (150,)
        assert len(np.unique(y)) == 3
    elif str_dataset == 'breast-cancer':
        path = 'data/breast-cancer-wisconsin.data'
        data = np.loadtxt(path, delimiter=',')
        X = data[:, 1:10]
        y = data[:, 10]
        assert X.shape == (699, 9)  # N*p
        assert y.shape == (699,)
        # BE CAREFUL: -1 are the missing values
        tmp = X[:, 5]
        mean = np.mean(tmp[tmp != -1])
        X[:, 5][X[:, 5] == -1] = mean
        assert (X >= 1).all()
        assert (X <= 10).all()
        assert len(np.unique(y)) == 2
    elif str_dataset == 'mnist':
        path = 'data/2k2k.mat'
        matstruct_contents = sio.loadmat(path)
        X = matstruct_contents['fea']
        y = matstruct_contents['gnd']
        y = y.reshape(-1)
        assert X.shape == (4000, 784)  # N*p
        assert y.shape == (4000,)
        assert len(np.unique(y)) == 10
    elif str_dataset == 'isolet':
        path = 'data/isolet.csv'
        data = np.genfromtxt(path, delimiter=' ')
        assert data.shape == (7797, 618)
        X = data[:, :617]
        y = data[:, 617]
        assert X.shape == (7797, 617)  # N*p
        assert y.shape == (7797,)
        assert len(np.unique(y)) == 26
    else:
        error = 'Dataset ' + str_dataset + ' not available...'
        raise ValueError(error)
    return X, y
