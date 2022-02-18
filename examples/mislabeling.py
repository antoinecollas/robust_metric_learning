from copy import deepcopy
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from metric_learn import Covariance, ITML_Supervised, LMNN
import os
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import time
import tikzplotlib
from tqdm import tqdm

from robust_metric_learning.data_loader import load_data
from robust_metric_learning.evaluation import\
        create_directory, clf_predict_evaluate
from robust_metric_learning.metric_learning import\
        Identity, GMML_Supervised, RGML


def main(
    random_state,
    datasets,
    n_runs,
    test_size,
    fractions_mislabeling,
    n_neighbors,
    clf,
    verbose
):
    matplotlib.use('Agg')
    path = create_directory('mislabeling')
    rnd.seed(random_state)

    def NUM_CONST(n_classes):
        return 75 * n_classes * (n_classes - 1)

    for dataset in datasets:
        start_time = time.time()

        if verbose >= 1:
            print()
            print('##############################')
            print('DATASET:', dataset)
            print('##############################')

        # load data
        X, y = load_data(dataset)
        _, p = X.shape
        n_classes = len(np.unique(y))
        num_constraints = NUM_CONST(n_classes)
        mean_errors_dict_frac_mislabels = dict()
        std_errors_dict_frac_mislabels = dict()

        for fraction_mislabeling in fractions_mislabeling:
            metrics_names = list()
            errors_dict = dict()

            if verbose >= 1:
                to_print = '##################### FRACTION MISLABELING:'
                to_print += str(fraction_mislabeling)
                to_print += ' #####################'
                print(to_print)

            iterator_n_runs = range(n_runs)
            if verbose >= 1:
                iterator_n_runs = tqdm(iterator_n_runs)

            for i in iterator_n_runs:
                # train test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state + i,
                    shuffle=True,
                    stratify=y
                )

                # create wrong labels
                new_y_train = deepcopy(y_train)
                frac_mislabel = np.sum(new_y_train != y_train) / len(y_train)
                while frac_mislabel < fraction_mislabeling:
                    idx = rnd.randint(len(y_train))
                    new_class = rnd.randint(np.max(y_train) + 1)
                    new_y_train[idx] = new_class
                    frac_mislabel = np.sum(new_y_train != y_train)
                    frac_mislabel /= len(y_train)
                y_train = new_y_train

                # ##########################
                # ##### UNSUPERVISED #######
                # ##########################

                # Euclidean
                metric_name = 'Euclidean'
                if verbose >= 2:
                    print('Metric name:', metric_name)
                pipe = Pipeline(
                    [(metric_name, Identity()), ('classifier', clf)])
                pipe.fit(X_train, y_train)
                clf_predict_evaluate(
                    X_test, y_test, metrics_names, metric_name,
                    pipe, errors_dict)

                # SCM
                metric_name = 'SCM'
                if verbose >= 2:
                    print('Metric name:', metric_name)
                pipe = Pipeline(
                    [(metric_name, Covariance()), ('classifier', clf)])
                pipe.fit(X_train, y_train)
                clf_predict_evaluate(
                    X_test, y_test, metrics_names, metric_name,
                    pipe, errors_dict)

                # ##########################
                # ### WEAKLY SUPERVISED ####
                # ##########################

                # ITML - identity
                metric_name = 'ITML - identity'
                if verbose >= 2:
                    print('Metric name:', metric_name)
                if (dataset in ['iris']) and (frac_mislabel > 0):
                    # the only way to make it work is to reduce gamma
                    gamma = 0
                else:
                    gamma = 1
                metric_learner = ITML_Supervised(
                    gamma=gamma,
                    num_constraints=num_constraints,
                    prior='identity',
                    random_state=random_state)
                pipe = Pipeline(
                    [(metric_name, metric_learner), ('classifier', clf)]
                )
                pipe.fit(X_train, y_train)
                clf_predict_evaluate(
                    X_test, y_test, metrics_names, metric_name,
                    pipe, errors_dict)

                # ITML - covariance
                metric_name = 'ITML - SCM'
                if verbose >= 2:
                    print('Metric name:', metric_name)
                metric_learner = ITML_Supervised(
                    gamma=gamma,
                    num_constraints=num_constraints,
                    prior='covariance',
                    random_state=random_state)
                pipe = Pipeline(
                    [(metric_name, metric_learner), ('classifier', clf)]
                )
                pipe.fit(X_train, y_train)
                clf_predict_evaluate(
                    X_test, y_test, metrics_names, metric_name,
                    pipe, errors_dict)

                # GMML
                if dataset in ['german', 'mnist', 'isolet']:
                    reg = 0.1
                else:
                    reg = 0

                metric_name = 'GMML - t=0'
                if verbose >= 2:
                    print('Metric name:', metric_name)
                metric_learner = GMML_Supervised(
                    regularization_param=reg, balance_param=0,
                    num_constraints=num_constraints,
                    random_state=random_state)
                pipe = Pipeline(
                    [(metric_name, metric_learner), ('classifier', clf)])
                pipe.fit(X_train, y_train)
                clf_predict_evaluate(
                    X_test, y_test, metrics_names, metric_name,
                    pipe, errors_dict)

                # ##########################
                # ####### SUPERVISED #######
                # ##########################

                # LMNN
                metric_name = 'LMNN'
                if verbose >= 2:
                    print('Metric name:', metric_name)
                metric_learner = LMNN(
                    k=n_neighbors, random_state=random_state)
                pipe = Pipeline(
                    [(metric_name, metric_learner), ('classifier', clf)]
                )
                pipe.fit(X_train, y_train)
                clf_predict_evaluate(
                    X_test, y_test, metrics_names, metric_name,
                    pipe, errors_dict)

                # #############################
                # ######### HOME MADE #########
                # #############################

                # RGML
                REG = 0.05

                def RGML_evaluate(metric_learner, metric_name):
                    pipe = Pipeline(
                        [(metric_name, metric_learner),
                         ('classifier', clf)])
                    pipe.fit(X_train, y_train)
                    clf_predict_evaluate(
                        X_test, y_test, metrics_names, metric_name,
                        pipe, errors_dict)

                metric_name_base = 'RGML'

                def rho(t):
                    return t

                metric_name = metric_name_base + '_Gaussian'
                metric_name += '_' + str(REG)
                if verbose >= 2:
                    print('Metric name:', metric_name)
                metric_learner = RGML(rho, divergence='Riemannian',
                                      regularization_param=REG,
                                      init='SCM', manifold='SPD',
                                      num_constraints=num_constraints,
                                      random_state=random_state)
                RGML_evaluate(metric_learner, metric_name)

                def rho(t):
                    return p * jnp.log(t)

                metric_name = metric_name_base + '_Tyler'
                metric_name += '_' + str(REG)
                if verbose >= 2:
                    print('Metric name:', metric_name)
                metric_learner = RGML(rho, divergence='Riemannian',
                                      regularization_param=REG,
                                      init='SCM', manifold='SSPD',
                                      num_constraints=num_constraints,
                                      random_state=random_state)
                RGML_evaluate(metric_learner, metric_name)

            # save results
            for metric_name in metrics_names:
                assert len(errors_dict[metric_name]) == n_runs
                if metric_name not in std_errors_dict_frac_mislabels:
                    mean_errors_dict_frac_mislabels[metric_name] = list()
                    std_errors_dict_frac_mislabels[metric_name] = list()
                mean_errors_dict_frac_mislabels[metric_name].append(
                    np.mean(errors_dict[metric_name]) * 100)
                std_errors_dict_frac_mislabels[metric_name].append(
                    np.std(errors_dict[metric_name]) * 100)

        elapsed_time = time.time() - start_time
        if verbose >= 1:
            print('Elapsed time (hh:mm:ss):',
                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        # plot and save
        x = np.array(fractions_mislabeling) * 100
        for key in mean_errors_dict_frac_mislabels:
            y = mean_errors_dict_frac_mislabels[key]
            plt.plot(x, y, label=key, marker='.')
        plt.legend(loc='upper left')
        plt.xlabel('Fraction of mislabeling (in %)')
        plt.ylabel('Classification error (in %)')
        plt.title('Dataset: ' + dataset)
        path_fig = os.path.join(path, 'results_' + dataset)
        plt.savefig(path_fig + '.png')
        tikzplotlib.save(path_fig + '.tex')
        plt.close('all')

        # table and save
        tmp = ['Methods']
        for mislabel_rate in x:
            tmp.append(str(mislabel_rate) + '%')
        t = PrettyTable(tmp)
        t.title = 'Mean error (in %) with respect to mislabeling rate'
        for key in mean_errors_dict_frac_mislabels:
            y = mean_errors_dict_frac_mislabels[key]
            row = [key]
            for value in y:
                row.append(str(round(value, 2)))
            t.add_row(row)
        if verbose >= 1:
            print('Classification errors:')
            print(t)

        path_table = os.path.join(path, 'results_' + dataset + '.txt')
        with open(path_table, 'a') as f:
            print(t, file=f)


if __name__ == '__main__':
    RANDOM_STATE = 0
    DATASETS = ['iris', 'wine', 'vehicle']
    N_RUNS = 200
    TEST_SIZE = 0.5
    FRACTIONS_MISLABELING = [0, 0.05, 0.1, 0.15, 0.2]
    N_NEIGHBORS = 5
    N_JOBS = -1
    CLF = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=N_JOBS)
    VERBOSE = 1

    main(
        random_state=RANDOM_STATE,
        datasets=DATASETS,
        n_runs=N_RUNS,
        test_size=TEST_SIZE,
        fractions_mislabeling=FRACTIONS_MISLABELING,
        n_neighbors=N_NEIGHBORS,
        clf=CLF,
        verbose=VERBOSE
    )
