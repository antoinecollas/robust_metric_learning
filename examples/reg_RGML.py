import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import tikzplotlib
from tqdm import tqdm

from robust_metric_learning.data_loader import load_data
from robust_metric_learning.evaluation import\
        create_directory, clf_predict_evaluate
from robust_metric_learning.metric_learning import RGML


def main(
    random_state,
    datasets,
    regs,
    n_runs,
    test_size,
    clf,
    verbose
):
    matplotlib.use('Agg')
    path = create_directory('reg_RGML')
    rnd.seed(random_state)

    for dataset in datasets:
        def NUM_CONST(n_classes):
            return 75 * n_classes * (n_classes - 1)

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
        mean_errors_dict_regs = dict()
        std_errors_dict_regs = dict()

        for reg in regs:
            metrics_names = list()
            errors_dict = dict()

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

                # RGML
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
                if verbose >= 2:
                    print('Metric name:', metric_name)

                metric_learner = RGML(rho, divergence='Riemannian',
                                      regularization_param=reg,
                                      init='SCM', manifold='SPD',
                                      num_constraints=num_constraints,
                                      random_state=random_state)
                RGML_evaluate(metric_learner, metric_name)

                def rho(t):
                    return p * jnp.log(t)

                metric_name = metric_name_base + '_Tyler'
                if verbose >= 2:
                    print('Metric name:', metric_name)

                metric_learner = RGML(rho, divergence='Riemannian',
                                      regularization_param=reg,
                                      init='SCM', manifold='SSPD',
                                      num_constraints=num_constraints,
                                      random_state=random_state)
                RGML_evaluate(metric_learner, metric_name)

            # save results
            for metric_name in metrics_names:
                assert len(errors_dict[metric_name]) == n_runs
                if metric_name not in std_errors_dict_regs:
                    mean_errors_dict_regs[metric_name] = list()
                    std_errors_dict_regs[metric_name] = list()
                mean_errors_dict_regs[metric_name].append(
                    np.mean(errors_dict[metric_name]) * 100)
                std_errors_dict_regs[metric_name].append(
                    np.std(errors_dict[metric_name]) * 100)

        # plot and save
        x = np.array(regs)
        for key in mean_errors_dict_regs:
            y = mean_errors_dict_regs[key]
            plt.semilogx(x, y, label=key, marker='.')
        plt.legend(loc='upper left')
        plt.xlabel('lambda (regularization)')
        plt.ylabel('Classification error (in %)')
        plt.title('Dataset: ' + dataset)
        path_fig = os.path.join(path, 'results_' + dataset)
        plt.savefig(path_fig + '.png')
        tikzplotlib.save(path_fig + '.tex')
        plt.close('all')


if __name__ == '__main__':
    RANDOM_STATE = 0
    DATASETS = ['wine', 'vehicle', 'iris']
    REGS = np.geomspace(start=5*1e-3, stop=5*1e-1, num=5)
    N_RUNS = 200
    TEST_SIZE = 0.5
    CLF = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    VERBOSE = 1

    main(
        random_state=RANDOM_STATE,
        datasets=DATASETS,
        regs=REGS,
        n_runs=N_RUNS,
        test_size=TEST_SIZE,
        clf=CLF,
        verbose=VERBOSE
    )
