import autograd.numpy.random as rnd


def create_S_D(X, y, num_const):
    k1 = rnd.randint(low=0, high=X.shape[0], size=num_const)
    k2 = rnd.randint(low=0, high=X.shape[0], size=num_const)
    ss = (y[k1] == y[k2])
    dd = (y[k1] != y[k2])
    S = X[k1[ss]] - X[k2[ss]]
    D = X[k1[dd]] - X[k2[dd]]
    return S, D
