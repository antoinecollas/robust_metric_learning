import jax.numpy as jnp
import numpy as np
import os
from metric_learn import Covariance, ITML_Supervised, LMNN
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from robust_metric_learning.data_loader import load_data
from robust_metric_learning.evaluation import\
        create_directory, clf_predict_evaluate
from robust_metric_learning.metric_learning import\
        Identity, GMML_Supervised, MeanSCM, RML, SPDMeanSCM


def main(
    random_state,
    datasets,
    n_runs,
    test_size,
    n_cv_grid_search,
    n_neighbors,
    n_jobs,
    clf,
    verbose
):
    path = create_directory('comparison')

    for dataset in datasets:
        if dataset in ['mnist', 'isolet']:
            def NUM_CONST(n_classes):
                return 200 * n_classes * (n_classes - 1)
        else:
            def NUM_CONST(n_classes):
                return 40 * n_classes * (n_classes - 1)

        if verbose >= 1:
            print('##############################')
            print('DATASET:', dataset)
            print('##############################')

        # load data
        X, y = load_data(dataset)
        _, p = X.shape
        n_classes = len(np.unique(y))
        num_constraints = NUM_CONST(n_classes)
        classif_errors_dict = dict()
        metrics_names = list()

        iterator = range(n_runs)
        if verbose >= 1:
            iterator = tqdm(iterator)

        for i in iterator:
            # train test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state + i,
                shuffle=True,
                stratify=y
            )

            # ##########################
            # ##### UNSUPERVISED #######
            # ##########################

            # Euclidean
            metric_name = 'Euclidean'
            if verbose >= 2:
                print('Metric name:', metric_name)
            pipe = Pipeline([(metric_name, Identity()), ('classifier', clf)])
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(
                X_test, y_test, metrics_names, metric_name,
                pipe, classif_errors_dict)

            # SCM
            metric_name = 'SCM'
            if verbose >= 2:
                print('Metric name:', metric_name)
            pipe = Pipeline([(metric_name, Covariance()), ('classifier', clf)])
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(
                X_test, y_test, metrics_names, metric_name,
                pipe, classif_errors_dict)

            # ##########################
            # ### WEAKLY SUPERVISED ####
            # ##########################

            # ITML - identity
            metric_name = 'ITML - identity'
            if verbose >= 2:
                print('Metric name:', metric_name)
            metric_learner = ITML_Supervised(
                prior='identity', num_constraints=num_constraints,
                random_state=random_state)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(
                X_test, y_test, metrics_names, metric_name,
                pipe, classif_errors_dict)

            # GMML
            if dataset in ['german', 'mnist', 'isolet']:
                reg = 0.1
            else:
                reg = 0

            metric_name = 'GMML - t=0'
            if verbose >= 2:
                print('Metric name:', metric_name)
            metric_learner = GMML_Supervised(regularization_param=reg,
                                             balance_param=0,
                                             num_constraints=num_constraints,
                                             random_state=random_state)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                                 pipe, classif_errors_dict)

            if dataset in ['australian', 'isolet']:
                balance_param_grid = [0]
            else:
                balance_param_grid = [0, 0.1, 0.3, 0.5, 0.7, 0.9]

            metric_name = 'GMML - CV'
            if verbose >= 2:
                print('Metric name:', metric_name)
            metric_learner = GMML_Supervised(regularization_param=reg,
                                             num_constraints=num_constraints,
                                             random_state=random_state)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            param_grid = [{metric_name+'__balance_param': balance_param_grid}]
            grid_search_clf = GridSearchCV(
                pipe, param_grid, cv=n_cv_grid_search,
                scoring=make_scorer(accuracy_score),
                refit=True, n_jobs=n_jobs)
            grid_search_clf.fit(X_train, y_train)
            pipe = grid_search_clf.best_estimator_
            if verbose >= 2:
                print('GMML cross val best param:')
                print(grid_search_clf.best_params_)
            clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                                 pipe, classif_errors_dict)

            # ##########################
            # ####### SUPERVISED #######
            # ##########################

            # LMNN
            metric_name = 'LMNN'
            if verbose >= 2:
                print('Metric name:', metric_name)
            metric_learner = LMNN(k=n_neighbors, random_state=random_state)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(
                X_test, y_test, metrics_names, metric_name,
                pipe, classif_errors_dict)

            # #############################
            # ######### HOME MADE #########
            # #############################

            # Mean SCM
            metric_name = 'Mean - SCM'
            if verbose >= 2:
                print('Metric name:', metric_name)

            if dataset == 'mnist':
                reg = 0.1
            else:
                reg = 0

            metric_learner = MeanSCM(
                regularization_param=reg, random_state=random_state)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                                 pipe, classif_errors_dict)

            # SPDMeanSCM
            if dataset in ['german', 'australian', 'breast-cancer']:
                reg = 0.1
            else:
                reg = 0

            metric_name = 'SPDMeanSCM'
            metric_learner = SPDMeanSCM(
                regularization_param=reg,
                num_constraints=num_constraints,
                random_state=random_state)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(
                X_test, y_test, metrics_names, metric_name,
                pipe, classif_errors_dict)

            # RML
            def RML_evaluate(metric_learner, metric_name):

                pipe = Pipeline(
                    [(metric_name, metric_learner), ('classifier', clf)]
                )
                pipe.fit(X_train, y_train)
                clf_predict_evaluate(
                    X_test, y_test, metrics_names, metric_name,
                    pipe, classif_errors_dict)

            metric_name_base = 'RML'

            # ###################
            # Gaussian
            # ###################

            def rho(t):
                return t

            for reg in [1, 0.1, 0.01]:
                metric_name = metric_name_base + '_Gaussian'
                metric_name += '_' + str(reg)
                if verbose >= 2:
                    print('Metric name:', metric_name)
                metric_learner = RML(rho, divergence='Riemannian',
                                     regularization_param=reg,
                                     init='SCM', manifold='SPD',
                                     random_state=123)
                RML_evaluate(metric_learner, metric_name)

            # ###################
            # Tyler
            # ###################

            def rho(t):
                return p * jnp.log(t)

            for reg in [1, 0.1, 0.01]:
                metric_name = metric_name_base + '_Tyler'
                metric_name += '_' + str(reg)
                if verbose >= 2:
                    print('Metric name:', metric_name)
                metric_learner = RML(rho, divergence='Riemannian',
                                     regularization_param=reg,
                                     init='SCM', manifold='SSPD',
                                     random_state=123)
                RML_evaluate(metric_learner, metric_name)

        t = PrettyTable(['Method', 'Mean error', 'std'])
        for metric_name in metrics_names:
            mean_error = np.mean(classif_errors_dict[metric_name]) * 100
            std_error = np.std(classif_errors_dict[metric_name]) * 100
            t.add_row([metric_name,
                       str(round(mean_error, 2)),
                       str(round(std_error, 2))])
        if verbose >= 1:
            print('Classification errors:')
            print(t)

        path_table = os.path.join(path, 'results_' + dataset + '.txt')
        with open(path_table, "a") as f:
            print(t, file=f)


if __name__ == '__main__':
    RANDOM_STATE = 0
    N_RUNS = 40
    TEST_SIZE = 0.5
    N_CV_GRID_SEARCH = 5
    N_NEIGHBORS = 5
    N_JOBS = -1
    CLF = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=N_JOBS)
    VERBOSE = 1

    # SMALL_DATASETS = True
    # if SMALL_DATASETS:
    #     DATASETS = ['wine', 'pima', 'vehicle']
    #     DATASETS = DATASETS + ['australian', 'iris']
    # else:
    #     DATASETS = ['mnist', 'isolet', 'letters']
    DATASETS = ['wine', 'vehicle', 'iris']

    main(
        random_state=RANDOM_STATE,
        datasets=DATASETS,
        n_runs=N_RUNS,
        test_size=TEST_SIZE,
        n_cv_grid_search=N_CV_GRID_SEARCH,
        n_neighbors=N_NEIGHBORS,
        n_jobs=N_JOBS,
        clf=CLF,
        verbose=VERBOSE
    )
