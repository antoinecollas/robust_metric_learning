import autograd
import autograd.numpy as np
import autograd.numpy.linalg as la
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import HermitianPositiveDefinite
# from pymanopt.solvers import ConjugateGradient, SteepestDescent
from pymanopt.solvers import ConjugateGradient

from matrix_operators import powm


# ICML 2016 "Geometric Mean Metric Learning" Zadeh et al.

def SPD_mean(A, B, t=0.5):
    assert t >= 0
    assert t <= 1
    A_sqrt = powm(A, 0.5)
    A_invsqrt = powm(A, -0.5)
    mean = A_sqrt@powm(A_invsqrt@B@A_invsqrt, t)@A_sqrt
    return mean


def GMML(S_train, D_train, t):
    S = S_train.T@S_train
    D = D_train.T@D_train

    S_inv = powm(S, -1)
    A = SPD_mean(S_inv, D, t)

    return A


# robust metric learning

def create_cost_egrad(S_train, D_train, rho):
    @pymanopt.function.Callable
    def cost(A):
        Q = np.real(np.einsum('ij,ji->i', S_train@A, S_train.T))
        res = np.mean(rho(Q))
        Q = np.real(np.einsum('ij,ji->i', D_train@la.inv(A), D_train.T))
        res = res + np.mean(rho(Q))
        return res

    @pymanopt.function.Callable
    def auto_egrad(A):
        res = autograd.grad(cost)(A)
        return res

    return cost, auto_egrad


def RBL(S_train, D_train, rho):
    p = S_train.shape[1]
    init = np.eye(p)

    cost, egrad = create_cost_egrad(S_train, D_train, rho)
    manifold = HermitianPositiveDefinite(p)
    # solver = SteepestDescent(logverbosity=2)
    solver = ConjugateGradient(maxiter=1e4, minstepsize=1e-10, logverbosity=2)
    problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=10)
    A, _ = solver.solve(problem, x=init)
    return A
