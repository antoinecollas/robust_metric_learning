import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import numpy as np
import numpy.random as rnd
from metric_learn.base_metric import MahalanobisMixin
from metric_learn.constraints import Constraints, wrap_pairs
from metric_learn._util import components_from_metric
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import (HermitianPositiveDefinite,
                                SpecialHermitianPositiveDefinite)
from pymanopt.solvers import ConjugateGradient, SteepestDescent
from sklearn.base import TransformerMixin

from .matrix_operators import logm, powm

jax.config.update('jax_enable_x64', True)


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

        self.components_ = powm(A, 0.5)

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

        return _BaseGMML._fit(self, pairs, y)


# Mean SCM


class MeanSCM(MahalanobisMixin, TransformerMixin):
    def __init__(self, regularization_param=0,
                 num_constraints=None, preprocessor=None,
                 random_state=None):
        super(MeanSCM, self).__init__(preprocessor)
        self.regularization_param = regularization_param
        self.num_constraints = num_constraints
        self.random_state = random_state

    def fit(self, X, y):
        random_state = self.random_state
        reg = self.regularization_param
        num_constraints = self.num_constraints

        rnd.seed(random_state)
        X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
        num_classes = len(np.unique(y))

        if num_constraints is None:
            num_constraints = 40 * num_classes * (num_classes - 1)

        N, p = X.shape
        A = np.zeros_like(X.T @ X)
        classes = np.unique(y)
        num_constraints_per_class = int(num_constraints / num_classes)

        for k in classes:
            mask = (y == k)

            # compute proportion of k-th class
            pi_k = np.sum(mask) / N

            # create positive pairs for the k-th class
            X_k = X[mask, :]
            a = rnd.randint(X_k.shape[0], size=num_constraints_per_class)
            b = rnd.randint(X_k.shape[0], size=num_constraints_per_class)

            # compute SCM on the similarity vectors
            tmp = X_k[a] - X_k[b]
            S_k = (1 / tmp.shape[0]) * (tmp.T @ tmp)
            X_k = X[y == k, :]

            A = A + pi_k * S_k

        A = A + reg * np.eye(p)
        A = powm(A, -1)
        self.components_ = components_from_metric(np.atleast_2d(A))

        return self


# SPD mean SCM


def _create_cost_egrad(pi, S):
    S_invsqrt = powm(S, -0.5)

    @pymanopt.function.Callable
    def cost(A):
        tmp = logm(S_invsqrt @ A @ S_invsqrt)
        tmp = jla.norm(tmp, axis=(1, 2))**2
        res = pi * tmp
        res = jnp.sum(res)
        return res

    @pymanopt.function.Callable
    def auto_egrad(A):
        res = jax.grad(cost)(A)
        return res

    return cost, auto_egrad


class SPDMeanSCM(MahalanobisMixin, TransformerMixin):
    def __init__(self, regularization_param=0,
                 num_constraints=None, preprocessor=None,
                 random_state=None):
        super(SPDMeanSCM, self).__init__(preprocessor)
        self.regularization_param = regularization_param
        self.num_constraints = num_constraints
        self.random_state = random_state

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : (n x d) matrix
          Input data, where each row corresponds to a single instance.
        y : (n) array-like
          Data labels.
        """
        random_state = self.random_state
        reg = self.regularization_param
        num_constraints = self.num_constraints

        rnd.seed(random_state)
        X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
        num_classes = len(np.unique(y))

        if num_constraints is None:
            num_constraints = 40 * num_classes * (num_classes - 1)

        classes = np.unique(y).astype(int)
        K = len(classes)
        N, p = X.shape
        pi = np.zeros(K)
        num_constraints_per_class = int(num_constraints / num_classes)
        S = np.zeros((K, p, p))

        for k in classes:
            mask = (y == k)

            # compute proportion of k-th class
            pi[k] = np.sum(mask) / N

            # create positive pairs for the k-th class
            X_k = X[mask, :]
            a = rnd.randint(X_k.shape[0], size=num_constraints_per_class)
            b = rnd.randint(X_k.shape[0], size=num_constraints_per_class)

            # compute SCM on the similarity vectors
            tmp = X_k[a] - X_k[b]
            S_k = (1 / tmp.shape[0]) * (tmp.T @ tmp)
            S[k, :, :] = S_k + reg * np.eye(p)

        # cost
        cost, egrad = _create_cost_egrad(pi, S)

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
        A = powm(A, -1)

        # store
        self.components_ = components_from_metric(np.atleast_2d(A))

        return self


# RGML: Robust Geometric Metric Learning
# Robust pooled covariance matrix


def _create_cost_egrad_RGML(rho, divergence, pi, X, reg):
    p = X.shape[1]

    def squared_Riemannian_distance(A, B):
        B_invsqrt = powm(B, -0.5)
        tmp = logm(B_invsqrt @ A @ B_invsqrt)
        d = jla.norm(tmp, axis=(1, 2)) ** 2
        return d

    def KL_left(A, B):
        C = powm(A, -1) @ B
        d = jnp.trace(C, axis1=-2, axis2=-1) - jnp.log(jla.det(C))
        return d

    def KL_right(A, B):
        return KL_left(B, A)

    def ellipticity_left(A, B):
        C = powm(A, -1) @ B
        d = p * jnp.log(jnp.trace(C, axis1=-2, axis2=-1))
        d = d - jnp.log(jla.det(C))
        return d

    def ellipticity_right(A, B):
        return ellipticity_left(B, A)

    if divergence == 'Riemannian':
        div = squared_Riemannian_distance
    elif divergence == 'KL-right':
        div = KL_right
    elif divergence == 'KL-left':
        div = KL_left
    elif divergence == 'ellipticity-right':
        div = ellipticity_right
    elif divergence == 'ellipticity-left':
        div = ellipticity_left
    else:
        raise ValueError('Divergence not implemented: ' + divergence)

    @pymanopt.function.Callable
    def cost(params):
        # likelihoods computation
        cov = params[1:, :, :]
        cov_inv = powm(cov, -1)
        Q = jnp.real(jnp.einsum('lij,ljk,lik->li', X, cov_inv, X))
        L = jnp.mean(rho(Q), axis=1) + jnp.log(jnp.real(jla.det(cov)))

        # distances computation
        A = params[0, :, :]
        d = div(A, cov)

        # regularized likelihood
        tmp = pi * (L + reg * d)
        L_reg = jnp.sum(tmp)

        return L_reg

    @pymanopt.function.Callable
    def auto_egrad(params):
        # print(jnp.real(jla.eigvals(params[0, :, :])))
        grad = jax.grad(cost)(params)
        return grad

    return cost, auto_egrad


class RGML(MahalanobisMixin, TransformerMixin):
    def __init__(self, rho=None, divergence='Riemannian',
                 regularization_param=0.1,
                 init='SCM', manifold='SPD',
                 solver='ConjugateGradient',
                 maxiter=1e3,
                 minstepsize=1e-10,
                 mingradnorm=1e-3,
                 num_constraints=None, preprocessor=None,
                 random_state=None):
        super(RGML, self).__init__(preprocessor)
        if rho is None:
            def rho(t):
                return t
        self.rho = rho
        self.divergence = divergence
        self.regularization_param = regularization_param
        self.init = init
        self.manifold = manifold
        self.solver = solver
        self.maxiter = maxiter
        self.minstepsize = minstepsize
        self.mingradnorm = mingradnorm
        self.num_constraints = num_constraints
        self.random_state = random_state

    def fit(self, X, y):
        """Create constraints from labels and learn the RGML model.
        Parameters
        ----------
        X : (n x d) matrix
          Input data, where each row corresponds to a single instance.
        y : (n) array-like
          Data labels.
        """
        rho = self.rho
        divergence = self.divergence
        reg = self.regularization_param
        num_constraints = self.num_constraints
        init = self.init
        manifold_name = self.manifold
        solver = self.solver
        maxiter = self.maxiter
        minstepsize = self.minstepsize
        mingradnorm = self.mingradnorm
        random_state = self.random_state

        rnd.seed(random_state)
        X, y = self._prepare_inputs(X, y, ensure_min_samples=2)
        num_classes = len(np.unique(y))

        if num_constraints is None:
            num_constraints = 40 * num_classes * (num_classes - 1)

        classes = np.unique(y).astype(int)
        K = len(classes)
        N, p = X.shape
        pi = np.zeros(K)
        num_constraints_per_class = int(num_constraints / num_classes)
        S = np.zeros((K, num_constraints_per_class, p))

        for k in classes:
            mask = (y == k)

            # compute proportion of k-th class
            pi[k] = np.sum(mask) / N

            # create positive pairs for the k-th class
            X_k = X[mask, :]
            a = rnd.randint(X_k.shape[0], size=num_constraints_per_class)
            b = rnd.randint(X_k.shape[0], size=num_constraints_per_class)

            # compute the x_i - x_j
            S[k, :, :] = X_k[a] - X_k[b]

            # make sure that the vectors of S[k, :, :]  \neq 0
            THRESHOLD = 1e-16
            mask = jla.norm(S[k, :, :], axis=1) < THRESHOLD
            while np.sum(mask) > 0:
                a = rnd.randint(X_k.shape[0], size=np.sum(mask))
                b = rnd.randint(X_k.shape[0], size=np.sum(mask))
                S[k, mask, :] = X_k[a] - X_k[b]
                mask = jla.norm(S[k, :, :], axis=1) < THRESHOLD

        # cost
        cost, egrad = _create_cost_egrad_RGML(rho, divergence, pi, S, reg)

        # manifold
        if manifold_name == 'SPD':
            manifold = HermitianPositiveDefinite(p, K + 1)
        elif manifold_name == 'SSPD':
            manifold = SpecialHermitianPositiveDefinite(p, K + 1)
        else:
            raise ValueError('Wrong manifold...')

        # initialization
        # BE CAREFUL: gradient of eigh can't be computed if
        # some eigenvalues are equal.
        # see: https://github.com/google/jax/issues/669#issuecomment-777052841
        # Hence a good initialization must be chosen...
        init_params = np.zeros((K + 1, p, p))
        if init == 'SCM':
            init_params[0, :, :] = np.cov(X.T)
            for k in range(1, init_params.shape[0]):
                X_k = X[y == (k - 1), :]
                init_params[k, :, :] = np.cov(X_k.T)
        elif init == 'random':
            for k in range(init_params.shape[0]):
                X = rnd.normal(size=(10 * p, p))
                init_params[k, :, :] = np.cov(X.T)
        else:
            raise ValueError('Wrong initialization...')

        # check condition number
        for k in range(init_params.shape[0]):
            tmp = init_params[k, :, :]
            c = jla.cond(tmp)
            if c > 1e6:
                tmp = tmp + 1e-3 * (np.trace(tmp) / p) * np.eye(p)
                init_params[k, :, :] = tmp

        # init with unit det if the manifold is SSPD
        if manifold_name == 'SSPD':
            for k in range(init_params.shape[0]):
                tmp = init_params[k, :, :]
                init_params[k, :, :] = tmp / (jla.det(tmp) ** (1 / p))

        # solver
        if solver == 'ConjugateGradient':
            solver = ConjugateGradient
        elif solver == 'SteepestDescent':
            solver = SteepestDescent
        else:
            raise ValueError('Wrong optimizer...')
        solver = solver(
            maxiter=maxiter, minstepsize=minstepsize,
            mingradnorm=mingradnorm, logverbosity=2)

        # solve
        problem = Problem(manifold=manifold, cost=cost,
                          egrad=egrad, verbosity=0)
        A, infos = solver.solve(problem, x=init_params)
        self.solver_infos = infos
        A = A[0, :, :]
        A = powm(A, -1)

        # store
        self.components_ = components_from_metric(np.atleast_2d(A))

        return self
