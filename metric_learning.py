import autograd
import autograd.numpy as np
import autograd.numpy.linalg as la
from metric_learn.base_metric import MahalanobisMixin
from metric_learn.constraints import Constraints, wrap_pairs
from metric_learn._util import components_from_metric
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import HermitianPositiveDefinite
# from pymanopt.solvers import ConjugateGradient, SteepestDescent
from pymanopt.solvers import ConjugateGradient
from sklearn.base import TransformerMixin

from matrix_operators import powm


# identity

class Identity(MahalanobisMixin, TransformerMixin):
    def __init__(self, preprocessor=None):
        super(Identity, self).__init__(preprocessor)

    def fit(self, X, y=None):
        X = self._prepare_inputs(X, ensure_min_samples=0)
        _, p = X.shape
        self.components_ = np.eye(p)
        return self

# SCM


class SCM(MahalanobisMixin, TransformerMixin):
    def __init__(self, preprocessor=None):
        super(SCM, self).__init__(preprocessor)

    def fit(self, X, y=None):
        X = self._prepare_inputs(X, ensure_min_samples=2)
        N, _ = X.shape
        mean = np.mean(X, axis=0, keepdims=True)
        X = X - mean
        A = (1 / N) * X.T @ X
        self.components_ = powm(A, -0.5)
        return self


# ICML 2016 "Geometric Mean Metric Learning" Zadeh et al.

def SPD_mean(A, B, t=0.5):
    assert t >= 0
    assert t <= 1
    A_sqrt = powm(A, 0.5)
    A_invsqrt = powm(A, -0.5)
    mean = A_sqrt @ powm(A_invsqrt @ B @ A_invsqrt, t) @ A_sqrt
    return mean


class _BaseGMML(MahalanobisMixin):
    """Geometric Mean Metric Learning (GMML)"""
    def __init__(self, balance_param, regularization_param,
                 preprocessor=None):
        super(_BaseGMML, self).__init__(preprocessor)
        self.balance_param = balance_param
        self.regularization_param = regularization_param

    def _fit(self, pairs, y, bounds=None):
        t = self.balance_param
        reg = self.regularization_param
        pairs, y = self._prepare_inputs(pairs, y,
                                        type_of_inputs='tuples')

        # S
        tmp = pairs[y == 1]
        S = tmp[:, 0, :] - tmp[:, 1, :]
        N, p = S.shape
        S = (1 / N) * S.T @ S
        S = S + reg * np.eye(p)

        # D
        tmp = pairs[y == -1]
        D = tmp[:, 0, :] - tmp[:, 1, :]
        N, p = D.shape
        D = (1 / N) * D.T @ D
        D = D + reg * np.eye(p)

        S_inv = powm(S, -1)
        A = SPD_mean(S_inv, D, t)

        self.components_ = components_from_metric(np.atleast_2d(A))
        return self


class GMML_Supervised(_BaseGMML, TransformerMixin):
    def __init__(self, balance_param=0.5, regularization_param=0,
                 num_constraints=None, preprocessor=None, random_state=None):
        _BaseGMML.__init__(self, balance_param, regularization_param,
                           preprocessor=preprocessor)
        self.random_state = random_state
        self.num_constraints = num_constraints

    def fit(self, X, y):
        """Create constraints from labels and learn the GMML model.
        Parameters
        ----------
        X : (n x d) matrix
          Input data, where each row corresponds to a single instance.
        y : (n) array-like
          Data labels.
        """
        X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
        num_constraints = self.num_constraints
        if num_constraints is None:
            num_classes = len(np.unique(y))
            num_constraints = 40 * num_classes * (num_classes - 1)

        c = Constraints(y)
        pos_neg = c.positive_negative_pairs(num_constraints,
                                            random_state=self.random_state)
        pairs, y = wrap_pairs(X, pos_neg)
        # check number of constraints
        N, _, _ = pairs[y == 1].shape
        assert N == num_constraints
        N, _, _ = pairs[y == -1].shape
        assert N == num_constraints

        return _BaseGMML._fit(self, pairs, y)


# robust metric learning

def _create_cost_egrad(S, D, rho, reg):
    N, p = S.shape

    @pymanopt.function.Callable
    def cost(A):
        Q = np.real(np.einsum('ij,ji->i', S @ A, S.T))
        res = np.mean(rho(Q)) - np.log(np.real(la.det(A)))
        return res

    @pymanopt.function.Callable
    def auto_egrad(A):
        res = autograd.grad(cost)(A)
        return res

    return cost, auto_egrad


class _BaseRML(MahalanobisMixin):
    """Robust Metric Learning (RML)"""
    def __init__(self, rho, regularization_param,
                 preprocessor=None, random_state=None):
        super(_BaseRML, self).__init__(preprocessor)
        self.rho = rho
        self.random_state = random_state
        self.regularization_param = regularization_param

    def _fit(self, pairs, y, bounds=None):
        pairs, y = self._prepare_inputs(pairs, y,
                                        type_of_inputs='tuples')

        # S
        tmp = pairs[y == 1]
        S = tmp[:, 0, :] - tmp[:, 1, :]

        # D
        tmp = pairs[y == -1]
        D = tmp[:, 0, :] - tmp[:, 1, :]

        # cost fct
        rho = self.rho
        if rho is None:
            def rho(t):
                return t
        reg = self.regularization_param
        cost, egrad = _create_cost_egrad(S, D, rho, reg)

        N, p = S.shape

        # manifold
        manifold = HermitianPositiveDefinite(p)

        # solve
        init = np.eye(p)
        solver = ConjugateGradient(
            maxiter=1e3, minstepsize=1e-10,
            mingradnorm=1e-4, logverbosity=2)
        problem = Problem(manifold=manifold, cost=cost,
                          egrad=egrad, verbosity=0)
        A, _ = solver.solve(problem, x=init)

        # store
        self.components_ = components_from_metric(np.atleast_2d(A))

        return self


class RML_Supervised(_BaseRML, TransformerMixin):
    def __init__(self, rho=None, regularization_param=0,
                 num_constraints=None, preprocessor=None, random_state=None):
        _BaseRML.__init__(self, rho, regularization_param,
                          preprocessor=preprocessor,
                          random_state=random_state)
        self.num_constraints = num_constraints

    def fit(self, X, y):
        """Create constraints from labels and learn the RML model.
        Parameters
        ----------
        X : (n x d) matrix
          Input data, where each row corresponds to a single instance.
        y : (n) array-like
          Data labels.
        """
        X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
        num_constraints = self.num_constraints
        if num_constraints is None:
            num_classes = len(np.unique(y))
            num_constraints = 40 * num_classes * (num_classes - 1)

        c = Constraints(y)
        pos_neg = c.positive_negative_pairs(num_constraints,
                                            random_state=self.random_state)
        pairs, y = wrap_pairs(X, pos_neg)
        # check number of constraints
        N, _, _ = pairs[y == 1].shape
        assert N == num_constraints
        N, _, _ = pairs[y == -1].shape
        assert N == num_constraints

        return _BaseRML._fit(self, pairs, y)
