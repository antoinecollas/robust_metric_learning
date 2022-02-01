# Inspired from: https://github.com/alexandrebarachant/pyRiemann/
# blob/master/pyriemann/utils/base.py
import autograd.numpy as np
import autograd.numpy.linalg as la


def _matrix_operator(Ci, operator):
    eigvals, eigvects = la.eigh(Ci)
    if Ci.ndim == 2:
        eigvals = operator(eigvals)[np.newaxis, ...]
        A = eigvects*eigvals
        B = np.transpose(np.conjugate(eigvects))
        return A@B
    else:
        eigvals = operator(eigvals)[:, np.newaxis, ...]
        A = eigvects*eigvals
        B = np.transpose(np.conjugate(eigvects), axes=(0, 2, 1))
        C = np.einsum('ijk,ikl->ijl', A, B)
        return C


def logm(Ci):
    return _matrix_operator(Ci, np.log)


def expm(Ci):
    return _matrix_operator(Ci, np.exp)


def powm(Ci, alpha):
    def power(x):
        return x**alpha
    return _matrix_operator(Ci, power)
