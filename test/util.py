import numpy as np
import scipy.linalg as linalg


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
