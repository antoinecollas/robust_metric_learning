import autograd.numpy as np
import autograd.numpy.random as rnd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from matrix_operators import powm

# constants
SEED = 10
t_CONST = 0
N_JOBS = -1


def NUM_CONST(n_classes):
    return 40*n_classes*(n_classes-1)


# SPD mean

def SPD_mean(A, B, t=0.5):
    assert t >= 0
    assert t <= 1
    A_sqrt = powm(A, 0.5)
    A_invsqrt = powm(A, -0.5)
    mean = A_sqrt@powm(A_invsqrt@B@A_invsqrt, t)@A_sqrt
    return mean


# load data
data = load_wine()
X = data.data
y = data.target
assert type(X) is np.ndarray
assert type(y) is np.ndarray
assert X.shape == (178, 13)  # N*p
assert y.shape == (178,)
n_classes = len(np.unique(y))

# train test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=SEED)

# similarity, dissimilarity
k1 = rnd.randint(low=0, high=X_train.shape[0], size=NUM_CONST(n_classes))
k2 = rnd.randint(low=0, high=X_train.shape[0], size=NUM_CONST(n_classes))
ss = (y[k1] == y[k2])
dd = (y[k1] != y[k2])
S_train = X[k1[ss]] - X[k2[ss]]
D_train = X[k1[dd]] - X[k2[dd]]

# ICML 2016 "Geometric Mean Metric Learning" Zadeh et al.
S = S_train.T@S_train
D = D_train.T@D_train

S_inv = powm(S, -1)
A = SPD_mean(S_inv, D, t_CONST)
A_sqrt = powm(A, 0.5)

# k-nn
X_train = X_train@A_sqrt
X_test = X_test@A_sqrt
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# error
error = np.mean(y_pred != y_test)
print(error)
