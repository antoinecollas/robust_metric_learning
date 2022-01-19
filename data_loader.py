import autograd.numpy as np
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
    else:
        error = 'Dataset ' + str_dataset + ' not available...'
        raise ValueError(error)
    return X, y
