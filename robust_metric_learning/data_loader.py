import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.datasets import load_iris, load_wine


def load_data(str_dataset):
    if str_dataset == 'wine':
        data = load_wine()
        X = data.data
        y = data.target
    elif str_dataset == 'pima':
        path = 'data/pima.csv'
        data = np.genfromtxt(path, delimiter=' ')
        X = data[:, :8]
        y = data[:, 8]
    elif str_dataset == 'vehicle':
        path = 'data/vehicle.csv'
        data = np.genfromtxt(path, delimiter=' ')
        X = data[:, :18]
        y = data[:, 18]
    elif str_dataset == 'german':
        path = 'data/german.csv'
        data = np.genfromtxt(path, delimiter=',')
        X = data[:, :24]
        y = data[:, 24]
    elif str_dataset == 'australian':
        path = 'data/australian.mat'
        matstruct_contents = sio.loadmat(path)
        X = matstruct_contents['X']
        y = matstruct_contents['y']
        y = y.reshape(-1)
    elif str_dataset == 'iris':
        data = load_iris()
        X = data.data
        y = data.target
    elif str_dataset == 'breast-cancer':
        path = 'data/breast-cancer-wisconsin.data'
        data = np.loadtxt(path, delimiter=',')
        X = data[:, 1:10]
        y = data[:, 10]
        # BE CAREFUL: -1 are the missing values
        tmp = X[:, 5]
        mean = np.mean(tmp[tmp != -1])
        X[:, 5][X[:, 5] == -1] = mean
        assert (X >= 1).all()
        assert (X <= 10).all()
    elif str_dataset == 'segment':
        path = 'data/segmentation.txt'

        def convert_classes(s):
            s = str(s).lower()
            if s == 'brickface':
                return 0
            elif s == 'sky':
                return 1
            elif s == 'foliage':
                return 2
            elif s == 'cement':
                return 3
            elif s == 'window':
                return 4
            elif s == 'path':
                return 5
            elif s == 'grass':
                return 6
            else:
                return ValueError('Wrong classname...')

        # data = np.genfromtxt(path, converters=converters,
        #                      delimiter=',', skip_header=5)
        data = pd.read_csv(path, delimiter=',', skiprows=3)
        X = data.to_numpy()
        tmp = list(data.index)
        y = np.zeros(len(tmp))
        for i, s in enumerate(tmp):
            y[i] = convert_classes(s)
    elif str_dataset == 'mnist':
        path = 'data/2k2k.mat'
        matstruct_contents = sio.loadmat(path)
        X = matstruct_contents['fea']
        y = matstruct_contents['gnd']
        y = y.reshape(-1)
    elif str_dataset == 'isolet':
        path = 'data/isolet.csv'
        data = np.genfromtxt(path, delimiter=' ')
        X = data[:, :617]
        y = data[:, 617]
    elif str_dataset == 'letters':
        path = 'data/letters.csv'
        data = np.genfromtxt(path, delimiter=' ')
        X = data[:, :16]
        y = data[:, 16]
    else:
        error = 'Dataset ' + str_dataset + ' not available...'
        raise ValueError(error)

    # remap classes to 0, ..., K-1
    y = y.astype(int)
    classes = np.unique(y)
    for k, c in enumerate(classes):
        y[y == c] = k
    assert (np.unique(y) == np.arange(len(classes))).all()

    return X, y
