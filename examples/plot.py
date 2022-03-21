import numpy as np
import numpy.random as rnd
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.stats import ortho_group
import tikzplotlib

from robust_metric_learning.evaluation import create_directory
from robust_metric_learning.matrix_operators import powm
from robust_metric_learning.metric_learning import RGML


def main(
    random_state,
    N,
    num_constraints,
    reg
):
    # seed
    rnd.seed(random_state)

    # matplotlib backend
    matplotlib.use('Agg')

    # path
    path = create_directory('plot')

    # covariance generation
    Q = ortho_group.rvs(dim=2)
    D = np.diag([100, 1])
    cov = Q @ D @ Q.T

    # class 0
    mean_0 = np.array([0, 4])
    X_data_0 = rnd.multivariate_normal(mean=mean_0, cov=cov, size=N).T
    mean_0 = mean_0.reshape((-1, 1))

    # class 1
    mean_1 = np.array([0, -4])
    X_data_1 = rnd.multivariate_normal(mean=mean_1, cov=cov, size=N).T
    mean_1 = mean_1.reshape((-1, 1))

    # full data
    X = np.concatenate([X_data_0, X_data_1], axis=1)
    y = np.concatenate([np.zeros(X_data_0.shape[1]),
                        np.ones(X_data_1.shape[1])])

    # RGML
    def rho(t):
        return t
    metric_learner = RGML(rho, divergence='Riemannian',
                          regularization_param=reg,
                          init='SCM', manifold='SPD',
                          num_constraints=num_constraints,
                          random_state=random_state)
    metric_learner.fit(X.T, y)
    L = metric_learner.components_
    A = L.T @ L
    A_sqrt = powm(A, 0.5)

    MAX_PLT = 30

    # plot X
    plt.xlim(-MAX_PLT, MAX_PLT)
    plt.ylim(-MAX_PLT, MAX_PLT)
    plt.plot(X_data_0[0, :], X_data_0[1, :], '.', color='blue')
    plt.plot(X_data_1[0, :], X_data_1[1, :], '.', color='red')
    path_fig = os.path.join(path, 'raw_data')
    plt.savefig(path_fig + '.png')
    tikzplotlib.save(path_fig + '.tex')
    plt.close('all')

    # plot X whiten
    plt.xlim(-MAX_PLT, MAX_PLT)
    plt.ylim(-MAX_PLT, MAX_PLT)
    X_data_0 = A_sqrt @ X_data_0
    X_data_1 = A_sqrt @ X_data_1
    plt.plot(X_data_0[0, :], X_data_0[1, :], '.', color='blue')
    plt.plot(X_data_1[0, :], X_data_1[1, :], '.', color='red')
    path_fig = os.path.join(path, 'whiten_data')
    plt.savefig(path_fig + '.png')
    tikzplotlib.save(path_fig + '.tex')
    plt.close('all')


if __name__ == '__main__':
    RANDOM_STATE = 10
    N = 100
    NUM_CONSTRAINTS = 200
    REG = 0.01

    main(
        random_state=RANDOM_STATE,
        N=N,
        num_constraints=NUM_CONSTRAINTS,
        reg=REG
    )
