import autograd.numpy as np
import autograd.numpy.random as rnd
from functools import partial
from metric_learn import Covariance, ITML_Supervised, LMNN, MMC_Supervised
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from data_loader import load_data
from metric_learning import Identity, GMML_Supervised, RML_Supervised


# constants
SEED = 0
rnd.seed(SEED)
DATASETS = ['wine', 'breast-cancer', 'australian']
N_RUNS = 10
TEST_SIZE = 0.5
N_CV_GRID_SEARCH = 5
N_NEIGHBORS = 5
N_JOBS = 1
clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=N_JOBS)
FAST_TEST = True


def NUM_CONST(n_classes):
    return 40 * n_classes * (n_classes - 1)


def clf_predict_evaluate(X_test, y_test,
                         metrics_names, metric_name,
                         clf, classif_errors_dict):
    y_pred = clf.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    if metric_name not in metrics_names:
        metrics_names.append(metric_name)
        classif_errors_dict[metric_name] = list()
    classif_errors_dict[metric_name].append(error)


for dataset in DATASETS:
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

    for i in tqdm(range(N_RUNS)):
        # train test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=SEED + i,
            shuffle=True,
            stratify=y
        )

        # ##########################
        # ##### UNSUPERVISED #######
        # ##########################

        # Euclidean
        metric_name = 'Euclidean'
        pipe = Pipeline([(metric_name, Identity()), ('classifier', clf)])
        pipe.fit(X_train, y_train)
        clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                             pipe, classif_errors_dict)

        # SCM
        metric_name = 'SCM'
        pipe = Pipeline([(metric_name, Covariance()), ('classifier', clf)])
        pipe.fit(X_train, y_train)
        clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                             pipe, classif_errors_dict)

        # ##########################
        # ### WEAKLY SUPERVISED ####
        # ##########################

        if not FAST_TEST:
            # MMC
            metric_name = 'MMC'
            metric_learner = MMC_Supervised(
                num_constraints=num_constraints, random_state=SEED)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                                 pipe, classif_errors_dict)

            # ITML - identity
            metric_name = 'ITML - identity'
            metric_learner = ITML_Supervised(prior='identity',
                                             num_constraints=num_constraints,
                                             random_state=SEED)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                                 pipe, classif_errors_dict)

            # ITML - SCM
            metric_name = 'ITML - SCM'
            metric_learner = ITML_Supervised(
                prior='covariance', num_constraints=num_constraints,
                random_state=SEED)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                                 pipe, classif_errors_dict)

        # GMML
            reg = 0
        if dataset == 'australian':
            balance_param_grid = [0]
        else:
            balance_param_grid = [0, 0.25, 0.5, 0.75, 1]
        metric_name = 'GMML'
        metric_learner = GMML_Supervised(regularization_param=0,
                                         num_constraints=num_constraints,
                                         random_state=SEED)
        pipe = Pipeline([(metric_name, metric_learner), ('classifier', clf)])
        param_grid = [{'GMML__balance_param': balance_param_grid}]
        grid_search_clf = GridSearchCV(pipe, param_grid, cv=N_CV_GRID_SEARCH,
                                       scoring=make_scorer(accuracy_score),
                                       refit=True, n_jobs=N_JOBS)
        grid_search_clf.fit(X_train, y_train)
        pipe = grid_search_clf.best_estimator_
        clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                             pipe, classif_errors_dict)

        # ##########################
        # ####### SUPERVISED #######
        # ##########################

        if not FAST_TEST:
            # LMNN
            metric_name = 'LMNN'
            metric_learner = LMNN(k=N_NEIGHBORS, random_state=SEED)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                                 pipe, classif_errors_dict)

        # ##########################
        # ######### ROBUST #########
        # ##########################

        # RML
        def RML(rho, metric_name):
            metric_learner = RML_Supervised(rho, regularization_param=1e-8,
                                            num_constraints=num_constraints,
                                            random_state=SEED)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                                 pipe, classif_errors_dict)

        metric_name_base = 'RML'

        def rho_t(t):
            return t

        RML(rho_t, metric_name_base)

        def rho_log(t, c):
            return np.log(c + t)

        # C_TO_TEST = [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4]
        C_TO_TEST = [1]
        for c in C_TO_TEST:
            metric_name = metric_name_base + '_log_' + str(c)
            rho = partial(rho_log, c=c)
            RML(rho, metric_name)

        def rho_Huber(t, c):
            mask = t <= c
            res = mask * t
            res = res + (1 - mask) * c * (np.log(1e-10 + (t / c)) + 1)
            return res

        # C_TO_TEST = [1, 10, 1e2, 1e3, 1e4]
        C_TO_TEST = [1]
        for c in C_TO_TEST:
            metric_name = metric_name_base + '_Huber_' + str(c)
            rho = partial(rho_Huber, c=c)
            RML(rho, metric_name)

    print('Classification errors:')
    t = PrettyTable(['Method', 'Mean error', 'std'])
    for metric_name in metrics_names:
        mean_error = np.mean(classif_errors_dict[metric_name]) * 100
        std_error = np.std(classif_errors_dict[metric_name]) * 100
        t.add_row([metric_name,
                   str(round(mean_error, 2)), str(round(std_error, 2))])
    print(t)
