import autograd.numpy as np
import autograd.numpy.random as rnd
from metric_learn import Covariance, ITML_Supervised, LMNN, MMC_Supervised
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from data_loader import load_data
from metric_learning import\
        Identity,\
        GMML_Supervised,\
        MeanSCM,\
        RML,\
        SPDMeanSCM


# constants
SEED = 0
rnd.seed(SEED)

N_RUNS = 5
TEST_SIZE = 0.5
N_CV_GRID_SEARCH = 5
N_NEIGHBORS = 5
N_JOBS = -1
clf = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=N_JOBS)
FAST_TEST = True
ROBUST_METHODS = True
VERBOSE = False

SMALL_DATASETS = True
if SMALL_DATASETS:
    DATASETS = ['wine', 'pima', 'vehicle', 'german']
    DATASETS = DATASETS + ['australian', 'iris', 'breast-cancer']
else:
    DATASETS = ['mnist', 'isolet', 'letters']


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
    if dataset in ['mnist', 'isolet']:
        def NUM_CONST(n_classes):
            return 200 * n_classes * (n_classes - 1)
    else:
        def NUM_CONST(n_classes):
            return 40 * n_classes * (n_classes - 1)

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
        if VERBOSE:
            print('Metric name:', metric_name)
        pipe = Pipeline([(metric_name, Identity()), ('classifier', clf)])
        pipe.fit(X_train, y_train)
        clf_predict_evaluate(
            X_test, y_test, metrics_names, metric_name,
            pipe, classif_errors_dict)

        # SCM
        metric_name = 'SCM'
        if VERBOSE:
            print('Metric name:', metric_name)
        pipe = Pipeline([(metric_name, Covariance()), ('classifier', clf)])
        pipe.fit(X_train, y_train)
        clf_predict_evaluate(
            X_test, y_test, metrics_names, metric_name,
            pipe, classif_errors_dict)

        # ##########################
        # ### WEAKLY SUPERVISED ####
        # ##########################

        if not FAST_TEST:
            # MMC
            metric_name = 'MMC'
            if VERBOSE:
                print('Metric name:', metric_name)
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
            if VERBOSE:
                print('Metric name:', metric_name)
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
            if VERBOSE:
                print('Metric name:', metric_name)
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
        if dataset in ['german', 'mnist', 'isolet']:
            reg = 0.1
        else:
            reg = 0

        metric_name = 'GMML - t=0'
        if VERBOSE:
            print('Metric name:', metric_name)
        metric_learner = GMML_Supervised(regularization_param=reg,
                                         balance_param=0,
                                         num_constraints=num_constraints,
                                         random_state=SEED)
        pipe = Pipeline([(metric_name, metric_learner), ('classifier', clf)])
        pipe.fit(X_train, y_train)
        clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                             pipe, classif_errors_dict)

        if dataset in ['australian', 'isolet']:
            balance_param_grid = [0]
        else:
            balance_param_grid = [0, 0.1, 0.3, 0.5, 0.7, 0.9]

        metric_name = 'GMML - CV'
        if VERBOSE:
            print('Metric name:', metric_name)
        metric_learner = GMML_Supervised(regularization_param=reg,
                                         num_constraints=num_constraints,
                                         random_state=SEED)
        pipe = Pipeline([(metric_name, metric_learner), ('classifier', clf)])
        param_grid = [{metric_name+'__balance_param': balance_param_grid}]
        grid_search_clf = GridSearchCV(pipe, param_grid, cv=N_CV_GRID_SEARCH,
                                       scoring=make_scorer(accuracy_score),
                                       refit=True, n_jobs=N_JOBS)
        grid_search_clf.fit(X_train, y_train)
        pipe = grid_search_clf.best_estimator_
        if VERBOSE:
            print('GMML cross val best param:')
            print(grid_search_clf.best_params_)
        clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                             pipe, classif_errors_dict)

        # ##########################
        # ####### SUPERVISED #######
        # ##########################

        if not FAST_TEST:
            # LMNN
            metric_name = 'LMNN'
            if VERBOSE:
                print('Metric name:', metric_name)
            metric_learner = LMNN(k=N_NEIGHBORS, random_state=SEED)
            pipe = Pipeline(
                [(metric_name, metric_learner), ('classifier', clf)]
            )
            pipe.fit(X_train, y_train)
            clf_predict_evaluate(X_test, y_test, metrics_names, metric_name,
                                 pipe, classif_errors_dict)

        # #############################
        # ######### HOME MADE #########
        # #############################

        # Mean SCM
        metric_name = 'Mean - SCM'
        if VERBOSE:
            print('Metric name:', metric_name)

        if dataset == 'mnist':
            reg = 0.1
        else:
            reg = 0

        metric_learner = MeanSCM(regularization_param=reg)
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
            random_state=SEED)
        pipe = Pipeline(
            [(metric_name, metric_learner), ('classifier', clf)]
        )
        pipe.fit(X_train, y_train)
        clf_predict_evaluate(
            X_test, y_test, metrics_names, metric_name,
            pipe, classif_errors_dict)

        # RML
        if ROBUST_METHODS and (dataset not in ['mnist']):
            def RML_evaluate(rho, metric_name):
                metric_learner = RML(
                    rho, regularization_param=1e-8,
                    num_constraints=num_constraints,
                    random_state=SEED)
                pipe = Pipeline(
                    [(metric_name, metric_learner), ('classifier', clf)]
                )
                pipe.fit(X_train, y_train)
                clf_predict_evaluate(
                    X_test, y_test, metrics_names, metric_name,
                    pipe, classif_errors_dict)

            metric_name_base = 'RML'
            if VERBOSE:
                print('Metric name:', metric_name_base)

            def rho_t(t):
                return t

            RML_evaluate(rho_t, metric_name_base)

    print('Classification errors:')
    t = PrettyTable(['Method', 'Mean error', 'std'])
    for metric_name in metrics_names:
        mean_error = np.mean(classif_errors_dict[metric_name]) * 100
        std_error = np.std(classif_errors_dict[metric_name]) * 100
        t.add_row([metric_name,
                   str(round(mean_error, 2)), str(round(std_error, 2))])
    print(t)
