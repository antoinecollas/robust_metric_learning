import autograd.numpy as np
import scipy.io as sio
from sklearn.datasets import load_wine


def load_data(str_dataset):
    if str_dataset == 'wine':
        data = load_wine()
        X = data.data
        y = data.target
        assert X.shape == (178, 13)  # N*p
        assert y.shape == (178,)
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
    elif str_dataset == 'australian':
        path = 'data/australian.mat'
        matstruct_contents = sio.loadmat(path)
        X = matstruct_contents['X']
        y = matstruct_contents['y']
        y = y.reshape(-1)
        assert X.shape == (690, 14)  # N*p
        assert y.shape == (690,)
    elif str_dataset == 'mnist':
        path = 'data/2k2k.mat'
        matstruct_contents = sio.loadmat(path)
        X = matstruct_contents['fea']
        y = matstruct_contents['gnd']
        y = y.reshape(-1)
        assert X.shape == (4000, 784)  # N*p
        assert y.shape == (4000,)
    else:
        error = 'Dataset ' + str_dataset + ' not available...'
        raise ValueError(error)
    return X, y
