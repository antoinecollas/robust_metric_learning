import autograd.numpy as np
import autograd.numpy.random as rnd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from data_loader import load_data
from matrix_operators import powm
from metric_learning import GMML, RBL


# constants
SEED = 0
rnd.seed(SEED)
DATASET = 'breast-cancer'
GMML_REG = 0
RBL_REG = 0
N_JOBS = -1
N_RUNS = 10
TEST_SIZE = 0.5
N_NEIGHBORS = 5
CORRUPTED_PROPORTIONS = [0, 0.5, 1]


def NUM_CONST(n_classes):
    return 40*n_classes*(n_classes-1)


# load data
X, y = load_data(DATASET)
n_classes = len(np.unique(y))

for corrupted_proportion in CORRUPTED_PROPORTIONS:
    classif_error = dict()
    for i in range(N_RUNS):
        # train test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=SEED,
            shuffle=True,
            stratify=y
        )

        # similarity, dissimilarity
        n_constraints = NUM_CONST(n_classes)
        k1 = rnd.randint(low=0, high=X_train.shape[0], size=n_constraints)
        k2 = rnd.randint(low=0, high=X_train.shape[0], size=n_constraints)
        ss = (y[k1] == y[k2])
        dd = (y[k1] != y[k2])
        S_train = X[k1[ss]] - X[k2[ss]]
        D_train = X[k1[dd]] - X[k2[dd]]

        # corruption of S
        # generate a new D matrix to corrupt S
        k1 = rnd.randint(low=0, high=X_train.shape[0], size=n_constraints)
        k2 = rnd.randint(low=0, high=X_train.shape[0], size=n_constraints)
        dd = (y[k1] != y[k2])
        tmp_D_train = X[k1[dd]] - X[k2[dd]]
        n_corrupted = int(S_train.shape[0]*corrupted_proportion)
        k1 = rnd.randint(low=0, high=S_train.shape[0], size=n_corrupted)
        k2 = rnd.randint(low=0, high=tmp_D_train.shape[0], size=n_corrupted)
        S_train[k1] = tmp_D_train[k2]

        # corruption of D
        # generate a new S matrix to corrupt D
        k1 = rnd.randint(low=0, high=X_train.shape[0], size=n_constraints)
        k2 = rnd.randint(low=0, high=X_train.shape[0], size=n_constraints)
        ss = (y[k1] == y[k2])
        tmp_S_train = X[k1[ss]] - X[k2[ss]]
        n_corrupted = int(D_train.shape[0]*corrupted_proportion)
        k1 = rnd.randint(low=0, high=D_train.shape[0], size=n_corrupted)
        k2 = rnd.randint(low=0, high=tmp_S_train.shape[0], size=n_corrupted)
        D_train[k1] = tmp_S_train[k2]

        metrics = dict()

        # Euclidean
        _, p = X.shape
        metrics['Euclidean'] = np.eye(p)

        # GMML
        t_consts = [0, 0.5, 1]
        for t in t_consts:
            A = GMML(S_train, D_train, t, reg=GMML_REG)
            A_sqrt = powm(A, 0.5)
            metrics['GMML_' + str(t)] = A_sqrt

        # RBL
        def rho(t):
            return t
        A = RBL(S_train, D_train, rho, reg=RBL_REG)
        A_sqrt = powm(A, 0.5)
        metrics['RBL_t'] = A_sqrt

        def rho(t):
            return np.log(1e-15 + t)
        A = RBL(S_train, D_train, rho, reg=RBL_REG)
        A_sqrt = powm(A, 0.5)
        metrics['RBL_log_t'] = A_sqrt

        def rho(t):
            return np.log(1 + t)
        A = RBL(S_train, D_train, rho, reg=RBL_REG)
        A_sqrt = powm(A, 0.5)
        metrics['RBL_log_1+t'] = A_sqrt

        def rho(t):
            return np.log(1e-15 + t)
        A = RBL(S_train, D_train, rho, reg=RBL_REG)
        A_sqrt = powm(A, 0.5)
        metrics['RBL_log_t'] = A_sqrt

        for metric in metrics:
            A_sqrt = metrics[metric]

            # k-nn
            X_train_A = X_train@A_sqrt
            X_test_A = X_test@A_sqrt
            knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=N_JOBS)
            classifier = KNeighborsClassifier()
            classifier.fit(X_train_A, y_train)
            y_pred = classifier.predict(X_test_A)

            # error
            error = np.mean(y_pred != y_test)
            if metric not in classif_error:
                classif_error[metric] = list()
            classif_error[metric].append(error)

    print()
    print('Percentage of corrupted data:', corrupted_proportion*100, '%')
    print('Classification errors:')
    print('-------------------------------')
    for metric in metrics:
        mean_error = np.mean(classif_error[metric])*100
        std_error = np.std(classif_error[metric])*100
        str_print = metric
        str_print += ': ' + str(round(mean_error, 2)) + '%'
        str_print += ' +- ' + str(round(std_error, 2))
        print(str_print)
    print('-------------------------------')
