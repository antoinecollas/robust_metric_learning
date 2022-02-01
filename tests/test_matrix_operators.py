import jax
from jax.config import config
config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.numpy.linalg as jla
import numpy as np
import numpy.random as rnd
from numpy.testing import assert_allclose
import scipy.linalg as la

from robust_metric_learning.matrix_operators import (expm, logm, invm,
                                                     invsqrtm, powm)


def test_expm():
    rnd.seed(123)

    # tests with a single matrix
    # test 1
    C = 2 * np.eye(3)
    Ctrue = np.exp(2) * np.eye(3)
    assert_allclose(expm(C), Ctrue)

    # test 2
    Q, _ = la.qr(rnd.normal(size=(3, 3)))
    D = rnd.uniform(size=(3))
    A = Q @ np.diag(D) @ Q.T
    exp_A = Q @ np.diag(np.exp(D)) @ Q.T
    assert la.norm(expm(A) - exp_A) / la.norm(exp_A) < 0.01

    # test 3
    approx = np.eye(3) + A + (A @ A / 2)
    approx += (A @ A @ A / 6) + (A @ A @ A @ A / 24)
    assert la.norm(expm(A) - approx) / la.norm(approx) < 0.01

    # test on a batch of matrices
    A = np.zeros((10, 3, 3))
    exp_A = np.zeros((len(A), 3, 3))

    for i in range(len(A)):
        Q, _ = la.qr(rnd.normal(size=(3, 3)))
        D = rnd.uniform(size=(3))
        A[i] = Q @ np.diag(D) @ Q.T
        exp_A[i] = Q @ np.diag(np.exp(D)) @ Q.T

    assert expm(A).shape == (len(A), 3, 3)

    for i in range(len(A)):
        assert la.norm(expm(A)[i] - exp_A[i]) / la.norm(exp_A[i]) < 0.01


def test_logm():
    C = 2 * np.eye(3)
    Ctrue = np.log(2) * np.eye(3)
    assert_allclose(logm(C), Ctrue)


def test_invm():
    rnd.seed(123)
    p = 5
    K = 3

    # inversion
    C = 2 * np.eye(p)
    Ctrue = 0.5 * np.eye(p)
    assert_allclose(invm(C), Ctrue)

    # batch inversion
    X = rnd.normal(size=(K, 10 * p, p))
    x = rnd.normal(size=(p, 1))
    C = np.zeros((K, p, p))
    for k in range(K):
        C[k, :, :] = np.cov(X[k, :, :].T)
    assert_allclose(invm(C), jla.inv(C), rtol=1e-2)

    # grad computation
    X = rnd.normal(size=(10 * p, p))
    x = rnd.normal(size=(p, 1))
    C = np.cov(X.T)

    def cost(A):
        return np.real(x.T @ invm(A) @ x).reshape()
        # return np.real(x.T @ A @ x).reshape()
    grad = jax.grad(cost)(C)
    true_grad = - la.inv(C) @ (x @ x.T) @ la.inv(C)
    assert_allclose(grad, true_grad, rtol=1e-2)

    # batch grad computation
    X = rnd.normal(size=(K, 10 * p, p))
    x = rnd.normal(size=(K, p, 1))
    C = np.zeros((K, p, p))
    for k in range(K):
        C[k, :, :] = np.cov(X[k, :, :].T)

    def cost(C):
        return np.sum(np.real(jnp.transpose(x, axes=(0, 2, 1)) @ invm(C) @ x))
    grad = jax.grad(cost)(C)
    true_grad = np.zeros((K, p, p))
    for k in range(K):
        C_k = C[k, :, :]
        x_k = x[k, :, :]
        true_grad[k, :, :] = - la.inv(C_k) @ x_k @ x_k.T @ la.inv(C_k)
    assert_allclose(grad, true_grad, rtol=1e-2)


def test_invsqrtm():
    C = 2 * np.eye(3)
    Ctrue = (1.0 / np.sqrt(2)) * np.eye(3)
    assert_allclose(invsqrtm(C), Ctrue)


def test_powm():
    C = 2 * np.eye(3)
    Ctrue = (2 ** -1) * np.eye(3)
    assert_allclose(powm(C, -1), Ctrue)
