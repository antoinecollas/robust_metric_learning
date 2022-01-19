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
GMML_REG = 0
GMML_t_CONST_1 = 0
GMML_t_CONST_2 = 0.5
RBL_REG = 0
N_JOBS = -1
N_RUNS = 3
TEST_SIZE = 0.5
N_NEIGHBORS = 5
CORRUPTED_PROPORTION = 0


def NUM_CONST(n_classes):
    return 40*n_classes*(n_classes-1)


# load data
X, y = load_data('breast-cancer')
n_classes = len(np.unique(y))

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
    size = NUM_CONST(n_classes)
    k1 = rnd.randint(low=0, high=X_train.shape[0], size=size)
    k2 = rnd.randint(low=0, high=X_train.shape[0], size=size)
    ss = (y[k1] == y[k2])
    dd = (y[k1] != y[k2])
    S_train = X[k1[ss]] - X[k2[ss]]
    D_train = X[k1[dd]] - X[k2[dd]]

    # corruption of S
    # generate a new D matrix to corrupt S
    k1 = rnd.randint(low=0, high=X_train.shape[0], size=size)
    k2 = rnd.randint(low=0, high=X_train.shape[0], size=size)
    dd = (y[k1] != y[k2])
    tmp_D_train = X[k1[dd]] - X[k2[dd]]
    size = int(size*CORRUPTED_PROPORTION)
    k1 = rnd.randint(low=0, high=S_train.shape[0], size=size)
    k2 = rnd.randint(low=0, high=tmp_D_train.shape[0], size=size)
    S_train[k1] = tmp_D_train[k2]

    metrics = dict()

    # Euclidean
    _, p = X.shape
    metrics['Euclidean'] = np.eye(p)

    # GMML
    A = GMML(S_train, D_train, GMML_t_CONST_1, reg=GMML_REG)
    A_sqrt = powm(A, 0.5)
    metrics['GMML_' + str(GMML_t_CONST_1)] = A_sqrt

    A = GMML(S_train, D_train, GMML_t_CONST_2, reg=GMML_REG)
    A_sqrt = powm(A, 0.5)
    metrics['GMML_' + str(GMML_t_CONST_2)] = A_sqrt

    # RBL
    def rho(t):
        return t*t
    A = RBL(S_train, D_train, rho, reg=RBL_REG)
    A_sqrt = powm(A, 0.5)
    metrics['RBL'] = A_sqrt

    for metric in metrics:
        A_sqrt = metrics[metric]

        # k-nn
        X_train_A = X_train@A_sqrt
        X_test_A = X_test@A_sqrt
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=N_JOBS)
        knn.fit(X_train_A, y_train)
        y_pred = knn.predict(X_test_A)

        # error
        error = np.mean(y_pred != y_test)
        if metric not in classif_error:
            classif_error[metric] = list()
        classif_error[metric].append(error)

print('Percentage of corrupted data:', CORRUPTED_PROPORTION*100, '%')
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
