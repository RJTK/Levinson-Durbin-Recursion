import numpy as np
import scipy.linalg as linalg
# import numba


# TODO: Unfortunately this function uses a bunch of
# TODO: methods not supported by Numba.
# @numba.jit(nopython=False, cache=True)
def block_toeplitz(left_col, top_row=None):
    """
    Similarly to linalg.toeplitz but for blocks.
    """
    p = len(left_col)
    left_col = np.array(left_col)  # In case a list is passed in
    if top_row is None:
        top_row = [left_col[0]]
        top_row = np.array(top_row +
                           [np.transpose(np.conj(left_col[k]))
                            for k in range(1, p)])
    assert len(top_row) == p
    assert np.allclose(left_col[0], top_row[0])

    try:
        f = np.vstack((left_col, top_row[::-1][:-1]))
    except ValueError:
        # When the arrays are unidimensional
        f = np.hstack((left_col, top_row[::-1][:-1]))
    return np.block([[f[i - j] for j in range(p)] for i in range(p)])


def block_companion(B):
    """
    Produces a block companion from the matrices B[0], B[1], ... , B[p - 1]
    [B0, B1, B2, ... Bp-1]
    [ I,  0,  0, ... 0   ]
    [ 0,  I,  0, ... 0   ]
    [ 0,  0,  I, ... 0   ]
    [ 0,  0, ..., I, 0   ]
    """
    p = len(B)
    B = np.hstack([B[k] for k in range(p)])  # The top row
    n = B.shape[0]

    I = np.eye(n * (p - 1))
    Z = np.zeros((n * (p - 1), n))
    R = np.hstack((I, Z))
    B = np.vstack((B, R))
    return B


def system_rho(B):
    """
    Computes the syste stability coefficient for an all-pole
    system with coefficient matrices B[0], B[1], ...
    """
    C = block_companion(B)
    ev = linalg.eigvals(C)
    return max(abs(ev))


def is_stable(B):
    rho = system_rho(B)
    return rho < 1
