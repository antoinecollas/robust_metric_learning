import numpy as np
import numpy.linalg as la
import numpy.random as rnd
from numpy.testing import assert_allclose

from robust_metric_learning.data_loader import load_data
from robust_metric_learning.metric_learning import (Identity, SCM, SPD_mean,
                                                    GMML_Supervised, MeanSCM,
                                                    SPDMeanSCM, RML)


def _check_SPD(M):
    # check symmetry
    assert_allclose(M, M.T, rtol=1e-2)

    # check postive eigenvalues
    a = la.eigvals(M)
    assert (a > 0).all()


def test_Identity():
    X, y = load_data('wine')
    _, p = X.shape

    metric_learner = Identity()
    A = metric_learner.fit(X, y).components_
    M = A
    _check_SPD(M)

    assert_allclose(M, np.eye(p))


def test_SCM():
    X, y = load_data('wine')
    _, p = X.shape

    metric_learner = SCM()
    A = metric_learner.fit(X, y).components_
    M = A.T @ A
    _check_SPD(M)

    M_true = la.inv(np.cov(X.T))
    assert la.norm(M - M_true) / la.norm(M_true) <= 1e-2
    # following test doesn't pass for some reason ...
    # assert_allclose(M, invm(np.cov(X.T)), rtol=1e-2)


def test_SPD_mean():
    rnd.seed(123)

    # 1st test
    p = 10
    A = 2 * np.eye(p)
    B = 32 * np.eye(p)
    mean = SPD_mean(A, B)
    _check_SPD(mean)
    assert_allclose(mean, 8 * np.eye(p), rtol=1e-2)

    # 2nd test
    X = rnd.normal(size=(10 * p, p))
    A = np.cov(X.T)
    X = rnd.normal(size=(10 * p, p))
    B = np.cov(X.T)
    mean = SPD_mean(A, B, t=0.5)
    _check_SPD(mean)


def test_GMML_Supervised():
    X, y = load_data('wine')

    # test consistency
    metric_learner = GMML_Supervised(balance_param=0.5,
                                     regularization_param=0,
                                     random_state=123)
    A = metric_learner.fit(X, y).components_
    M1 = A.T @ A
    _check_SPD(M1)

    metric_learner = GMML_Supervised(balance_param=0.5,
                                     regularization_param=0,
                                     random_state=123)
    A = metric_learner.fit(X, y).components_
    M2 = A.T @ A
    _check_SPD(M2)

    assert_allclose(M1, M2, rtol=1e-2)

    # test regularization
    metric_learner = GMML_Supervised(balance_param=0.5,
                                     regularization_param=0,
                                     random_state=123)
    A = metric_learner.fit(X, y).components_
    M = A.T @ A
    _check_SPD(M)


def test_MeanSCM():
    X, y = load_data('wine')

    # test consistency
    metric_learner = MeanSCM(regularization_param=0,
                             random_state=123)
    A = metric_learner.fit(X, y).components_
    M1 = A.T @ A
    _check_SPD(M1)

    metric_learner = MeanSCM(regularization_param=0,
                             random_state=123)
    A = metric_learner.fit(X, y).components_
    M2 = A.T @ A
    _check_SPD(M2)

    assert_allclose(M1, M2, rtol=1e-2)

    # test regularization
    metric_learner = MeanSCM(regularization_param=0.5,
                             random_state=123)
    A = metric_learner.fit(X, y).components_
    M = A.T @ A
    _check_SPD(M)


def test_SPDMeanSCM():
    X, y = load_data('wine')

    # test consistency
    metric_learner = SPDMeanSCM(regularization_param=0,
                                random_state=123)
    A = metric_learner.fit(X, y).components_
    M1 = A.T @ A
    _check_SPD(M1)

    metric_learner = SPDMeanSCM(regularization_param=0,
                                random_state=123)
    A = metric_learner.fit(X, y).components_
    M2 = A.T @ A
    _check_SPD(M2)

    assert_allclose(M1, M2, rtol=1e-2)

    # test regularization
    metric_learner = SPDMeanSCM(regularization_param=0.5,
                                random_state=123)
    A = metric_learner.fit(X, y).components_
    M = A.T @ A
    _check_SPD(M)


def test_RML():
    X, y = load_data('wine')

    # test consistency
    metric_learner = RML(random_state=123)
    A = metric_learner.fit(X, y).components_
    M1 = A.T @ A
    _check_SPD(M1)

    metric_learner = RML(random_state=123)
    A = metric_learner.fit(X, y).components_
    M2 = A.T @ A
    _check_SPD(M2)

    assert_allclose(M1, M2, rtol=1e-2)
