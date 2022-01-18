# Inspired from: https://github.com/alexandrebarachant/pyRiemann/
# blob/master/pyriemann/utils/base.py
import autograd.numpy as np
import autograd.numpy.linalg as la


def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
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
    """Return the matrix logarithm of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the covariance matrix
    :returns: the matrix logarithm

    """
    return _matrix_operator(Ci, np.log)


def expm(Ci):
    """Return the matrix exponential of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the covariance matrix
    :returns: the matrix exponential

    """
    return _matrix_operator(Ci, np.exp)


def powm(Ci, alpha):
    """Return the matrix power :math:`\\alpha` of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha}
            \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the covariance matrix
    :param alpha: the power to apply
    :returns: the matrix power

    """
    def power(x):
        return x**alpha
    return _matrix_operator(Ci, power)



# the following is inspired by pymanopt: multi.py

def multiprod(A, B):
    """
    Inspired by MATLAB multiprod function by Paolo de Leva. A and B are
    assumed to be arrays containing M matrices, that is, A and B have
    dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each matrix
    in A with the corresponding matrix in B, using matrix multiplication.
    so multiprod(A, B) has dimensions (M, N, Q).
    """

    # First check if we have been given just one matrix
    if len(np.shape(A)) == 2:
        return np.dot(A, B)

    return np.einsum('ijk,ikl->ijl', A, B)

