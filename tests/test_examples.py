import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from examples.datasets import main as main_datasets
from examples.mislabeling import main as main_mislabeling
from examples.plot import main as main_plot
from examples.reg_RGML import main as main_reg_RGML
from examples.speed_convergence import main as main_speed_convergence


def test_datasets():
    DATASETS = ['wine']
    VERBOSE = 0

    main_datasets(datasets=DATASETS, verbose=VERBOSE)


def test_mislabeling():
    RANDOM_STATE = 0
    DATASETS = ['wine']
    N_RUNS = 2
    TEST_SIZE = 0.5
    FRACTIONS_MISLABELING = [0, 0.1]
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


def test_plot():
    RANDOM_STATE = 10
    N = 100
    NUM_CONSTRAINTS = 200
    REG = 0.01

    main_plot(
        random_state=RANDOM_STATE,
        N=N,
        num_constraints=NUM_CONSTRAINTS,
        reg=REG
    )


def test_speed_convergence():
    RANDOM_STATE = 0
    MAX_ITER = 5
    DATASETS = ['wine']
    VERBOSE = 0

    main_speed_convergence(
        random_state=RANDOM_STATE,
        maxiter=MAX_ITER,
        datasets=DATASETS,
        verbose=VERBOSE
    )


def test_reg_RGML():
    RANDOM_STATE = 0
    DATASETS = ['wine']
    REGS = np.geomspace(start=5*1e-3, stop=5*1e-1, num=3)
    N_RUNS = 2
    TEST_SIZE = 0.5
    CLF = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    VERBOSE = 0

    main_reg_RGML(
        random_state=RANDOM_STATE,
        datasets=DATASETS,
        regs=REGS,
        n_runs=N_RUNS,
        test_size=TEST_SIZE,
        clf=CLF,
        verbose=VERBOSE
    )
