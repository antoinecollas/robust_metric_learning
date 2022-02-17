import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tikzplotlib

from robust_metric_learning.data_loader import load_data
from robust_metric_learning.evaluation import create_directory
from robust_metric_learning.metric_learning import RGML


def plot_solver_infos(path_to_save, dataset_name,
                      solvers_infos, metric_names):
    # grad norm
    for solver_infos, metric_name in zip(solvers_infos, metric_names):
        x = solver_infos['iterations']['iteration']
        grad_norm = solver_infos['iterations']['gradnorm']
        plt.semilogy(x, grad_norm, label='grad norm ' + metric_name)
    plt.legend(loc='upper right')
    plt.xlabel('Number of iterations')
    plt.ylabel('Grad norm')
    path_fig = os.path.join(path_to_save, 'grad_norm_' + dataset_name)
    plt.savefig(path_fig + '.png')
    tikzplotlib.save(path_fig + '.tex')
    plt.close('all')

    # cost function
    for solver_infos, metric_name in zip(solvers_infos, metric_names):
        x = solver_infos['iterations']['iteration']
        tmp_min = np.min(solver_infos['iterations']['f(x)'])
        y = solver_infos['iterations']['f(x)'] - tmp_min + 1
        plt.semilogy(x, y, label='cost function ' + metric_name)
    plt.legend(loc='upper right')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost function')
    path_fig = os.path.join(path_to_save, 'cost_fct_' + dataset_name)
    plt.savefig(path_fig + '.png')
    tikzplotlib.save(path_fig + '.tex')
    plt.close('all')


def main(
    random_state,
    maxiter,
    datasets,
    verbose
):
    matplotlib.use('Agg')
    path = create_directory('speed_convergence')

    def NUM_CONST(n_classes):
        return 50 * n_classes * (n_classes - 1)

    for dataset in datasets:
        if verbose >= 1:
            print()
            print('##############################')
            print('DATASET:', dataset)
            print('##############################')
            print()

        # load data
        X, y = load_data(dataset)
        _, p = X.shape
        n_classes = len(np.unique(y))
        num_constraints = NUM_CONST(n_classes)

        if verbose >= 1:
            to_print = '##################### SPEED CONVERGENCE'
            to_print += ' #####################'
            print(to_print)

        # RGML
        metric_name_base = 'RGML'
        reg = 0.05

        if verbose >= 1:
            to_print = '##################### RGML Gaussian'
            to_print += ' #####################'
            print(to_print)

        def rho(t):
            return t

        metric_name = metric_name_base + '_Gaussian'
        metric_name += '_' + str(reg)
        if verbose >= 2:
            print('Metric name:', metric_name)
        metric_learner = RGML(rho, divergence='Riemannian',
                              regularization_param=reg,
                              init='SCM', manifold='SPD',
                              solver='SteepestDescent',
                              maxiter=maxiter,
                              minstepsize=0,
                              mingradnorm=0,
                              num_constraints=num_constraints,
                              random_state=random_state)
        metric_learner.fit(X, y)
        solver_infos_Gaussian = metric_learner.solver_infos

        if verbose >= 1:
            to_print = '##################### RGML Tyler'
            to_print += ' #####################'
            print(to_print)

        def rho(t):
            return p * jnp.log(t)

        metric_name = metric_name_base + '_Tyler'
        metric_name += '_' + str(reg)
        if verbose >= 2:
            print('Metric name:', metric_name)
        metric_learner = RGML(rho, divergence='Riemannian',
                              regularization_param=reg,
                              init='SCM', manifold='SSPD',
                              solver='SteepestDescent',
                              maxiter=maxiter,
                              minstepsize=0,
                              mingradnorm=0,
                              num_constraints=num_constraints,
                              random_state=random_state)
        metric_learner.fit(X, y)
        solver_infos_Tyler = metric_learner.solver_infos

        solvers_infos = [solver_infos_Gaussian, solver_infos_Tyler]
        metric_names = ['Gaussian', 'Tyler']
        plot_solver_infos(path, dataset, solvers_infos, metric_names)


if __name__ == '__main__':
    RANDOM_STATE = 0
    MAX_ITER = 40
    DATASETS = ['iris', 'wine', 'vehicle']
    VERBOSE = 1

    main(
        random_state=RANDOM_STATE,
        maxiter=MAX_ITER,
        datasets=DATASETS,
        verbose=VERBOSE
    )
