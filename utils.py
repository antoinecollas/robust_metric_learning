import autograd.numpy.random as rnd


def create_S_D(X, y, num_const):
    # generate two sets of indices
    k1 = rnd.randint(low=0, high=X.shape[0], size=num_const)
    k2 = rnd.randint(low=0, high=X.shape[0], size=num_const)

    # remove indices that are equal
    mask = ~(k1 == k2)
    k1 = k1[mask]
    k2 = k2[mask]
    assert (k1 != k2).all()

    # generate S and D
    ss = (y[k1] == y[k2])
    dd = (y[k1] != y[k2])
    S = X[k1[ss]] - X[k2[ss]]
    D = X[k1[dd]] - X[k2[dd]]

    return S, D
