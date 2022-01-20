import autograd.numpy.random as rnd


def create_S_D(X, y, num_const):
    k1 = rnd.randint(low=0, high=X.shape[0], size=num_const)
    k2 = rnd.randint(low=0, high=X.shape[0], size=num_const)
    ss = (y[k1] == y[k2])
    dd = (y[k1] != y[k2])
    S = X[k1[ss]] - X[k2[ss]]
    D = X[k1[dd]] - X[k2[dd]]

    # other way to create S and D
    # k1 = np.arange(X.shape[0])
    # k2 = np.arange(X.shape[0])
    # rnd.shuffle(k1)
    # rnd.shuffle(k2)
    # k1 = k1[:num_const]
    # k2 = k2[:num_const]
    # ss = (y[k1] == y[k2])
    # dd = (y[k1] != y[k2])
    # S = X[k1[ss]] - X[k2[ss]]
    # D = X[k1[dd]] - X[k2[dd]]
    return S, D
