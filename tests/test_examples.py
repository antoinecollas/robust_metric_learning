from sklearn.neighbors import KNeighborsClassifier

from examples.comparison import main as main_comparison
from examples.mislabeling import main as main_mislabeling


def test_comparison():
    RANDOM_STATE = 123
    DATASETS = ['wine']
    N_RUNS = 3
    TEST_SIZE = 0.5
    N_CV_GRID_SEARCH = 3
    N_NEIGHBORS = 5
    N_JOBS = 1
    CLF = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=N_JOBS)
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
        verbose=VERBOSE
    )


def test_mislabeling():
    RANDOM_STATE = 0
    DATASETS = ['wine']
    N_RUNS = 3
    TEST_SIZE = 0.5
    FRACTIONS_MISLABELING = [0, 0.05, 0.1]
    N_NEIGHBORS = 5
    N_JOBS = -1
    CLF = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=N_JOBS)
    VERBOSE = 0

    main_mislabeling(
        random_state=RANDOM_STATE,
        datasets=DATASETS,
        n_runs=N_RUNS,
        test_size=TEST_SIZE,
        fractions_mislabeling=FRACTIONS_MISLABELING,
        n_neighbors=N_NEIGHBORS,
        clf=CLF,
        verbose=VERBOSE
    )
