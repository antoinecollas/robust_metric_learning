# Inspired from: https://github.com/alexandrebarachant/pyRiemann/
# blob/master/pyriemann/utils/base.py
from jax.config import config
config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.numpy.linalg as jla


def _matrix_operator(Ci, operator):
    # eigh seems to be more stable than svd
    s, u = jla.eigh(Ci)
    # u, s, _ = jla.svd(Ci, full_matrices=False)
    if Ci.ndim == 2:
        s = operator(s)
        C = (u * s) @ jnp.transpose(u)
        return C
    else:
        s = operator(s)[:, jnp.newaxis, ...]
        A = u * s
        B = jnp.transpose(u, axes=(0, 2, 1))
        C = A @ B
        return C


def expm(Ci):
    return _matrix_operator(Ci, jnp.exp)


def logm(Ci):
    return _matrix_operator(Ci, jnp.log)


def invm(Ci):
    def inv_fct(x):
        return 1. / x
    return _matrix_operator(Ci, inv_fct)


def invsqrtm(Ci):
    def isqrt(x):
        return 1. / jnp.sqrt(x)
    return _matrix_operator(Ci, isqrt)


def powm(Ci, alpha):
    def power(x):
        return x ** alpha
    return _matrix_operator(Ci, power)
