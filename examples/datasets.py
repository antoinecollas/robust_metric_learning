import numpy as np

from robust_metric_learning.data_loader import load_data


def main(datasets, verbose):
    for dataset in datasets:
        if verbose >= 1:
            print()
            print('Dataset:', dataset)
        X, y = load_data(dataset)
        for k in np.unique(y):
            if verbose >= 1:
                y_k = y[y == k]
                print('Class', k, ':', len(y_k), 'samples')


if __name__ == '__main__':
    DATASETS = ['wine', 'vehicle', 'iris']
    VERBOSE = 1

    main(
        datasets=DATASETS,
        verbose=VERBOSE
    )
