from sklearn.neighbors import KNeighborsClassifier

from examples.comparison import main as main_comparison


def test_comparison():
    RANDOM_STATE = 123
    DATASETS = ['wine']
    N_RUNS = 3
    TEST_SIZE = 0.5
    N_CV_GRID_SEARCH = 3
    N_NEIGHBORS = 5
    N_JOBS = 1
    CLF = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=N_JOBS)
    FAST_TEST = True
    ROBUST_METHODS = True
    VERBOSE = 0

    main_comparison(
        random_state=RANDOM_STATE,
        datasets=DATASETS,
        n_runs=N_RUNS,
        test_size=TEST_SIZE,
        n_cv_grid_search=N_CV_GRID_SEARCH,
        n_neighbors=N_NEIGHBORS,
        n_jobs=N_JOBS,
        clf=CLF,
        fast_test=FAST_TEST,
        robust_methods=ROBUST_METHODS,
        verbose=VERBOSE
    )
