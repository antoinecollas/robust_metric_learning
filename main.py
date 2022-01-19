import autograd.numpy as np
import autograd.numpy.random as rnd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from matrix_operators import powm
from metric_learning import GMML, RBL


# constants
SEED = 0
t_CONST_GMML_1 = 0
t_CONST_GMML_2 = 0.5
N_JOBS = -1
N_RUNS = 1
TEST_SIZE = 0.5
N_NEIGHBORS = 5
rnd.seed(SEED)


def NUM_CONST(n_classes):
    return 40*n_classes*(n_classes-1)


# load data
data = load_wine()
X = data.data
y = data.target
print(type(X))
assert X.shape == (178, 13)  # N*p
assert y.shape == (178,)
n_classes = len(np.unique(y))

classif_error = dict()
for i in range(N_RUNS):
    # train test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)

    # similarity, dissimilarity
    k1 = rnd.randint(low=0, high=X_train.shape[0], size=NUM_CONST(n_classes))
    k2 = rnd.randint(low=0, high=X_train.shape[0], size=NUM_CONST(n_classes))
    ss = (y[k1] == y[k2])
    dd = (y[k1] != y[k2])
    S_train = X[k1[ss]] - X[k2[ss]]
    D_train = X[k1[dd]] - X[k2[dd]]

    metrics = dict()

    # Euclidean
    _, p = X.shape
    metrics['Euclidean'] = np.eye(p)

    # GMML
    A = GMML(S_train, D_train, t_CONST_GMML_1)
    A_sqrt = powm(A, 0.5)
    metrics['GMML_' + str(t_CONST_GMML_1)] = A_sqrt

    A = GMML(S_train, D_train, t_CONST_GMML_2)
    A_sqrt = powm(A, 0.5)
    metrics['GMML_' + str(t_CONST_GMML_2)] = A_sqrt

    # RBL
    def rho(t):
        return t
    A = RBL(S_train, D_train, rho)
    A_sqrt = powm(A, 0.5)
    metrics['RBL'] = A_sqrt
    # import numpy.linalg as la
    # print(la.norm(metrics['RBL'] - metrics['GMML'])/la.norm(metrics['GMML']))
    # print(la.norm(metrics['Euclidean'] -
    # metrics['GMML'])/la.norm(metrics['GMML']))

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

for metric in metrics:
    error = np.mean(classif_error[metric])
    print(metric, '- classification error:', round(error*100, 2), '%')
