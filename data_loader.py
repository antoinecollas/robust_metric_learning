from sklearn.datasets import load_wine


def load_data(str_dataset):
    if str_dataset == 'wine':
        data = load_wine()
        X = data.data
        y = data.target
        assert X.shape == (178, 13)  # N*p
        assert y.shape == (178,)
    else:
        error = 'Dataset ' + str_dataset + ' not available...'
        raise ValueError(error)
    return X, y
