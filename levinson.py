"""
Implementation of Levinson Recursion and other associated routines,
particularly the Block Toeplitz versions of Whittle and Akaike.
"""

import numpy as np
import numba


@numba.jit(nopython=True, cache=False)
def lev_durb(r):
    """
    Comsumes a length p + 1 vector r = [r(0), ..., r(p)]
    which must be a valid autocovariance sequence (i.e. positive
    semi-definite) and returns (a, G, eps) as follows:

    returns:
      - a (np.array): Length p + 1 array (with a[0] = 1.0) consisting
        of the filter coefficients for an all-pole model of a signal
        having autocovariance r.
      - G (np.array): Length p array of reflection coefficients.
        It is guaranteed that |G[tau]| <= 1.
      - eps (float): The error achieved by the all-pole model.  eps is
        guaranteed to satisfy eps >= 0.

    We get a solution to the system R @ a = eps * e1 where R = toep(r)
    and e1 is the first canonical basis vector.  The variables are a[1:]
    and eps.
    """
    # Initialization
    p = len(r) - 1
    a = np.zeros(p + 1)
    a[0] = 1.0
    G = np.zeros(p)
    eps = r[0]

    for tau in range(p):
        # Compute reflection coefficient
        conv = r[tau + 1]
        for s in range(1, tau + 1):
            conv = conv + a[s] * r[tau - s + 1]
        G[tau] = -conv / eps

        # Update 'a' vector
        a_cpy = np.copy(a)
        for s in range(1, tau + 1):
            a_cpy[s] = a[s] + G[tau] * np.conj(a[tau - s + 1])
        a = a_cpy
        a[tau + 1] = G[tau]
        eps = eps * (1 - np.abs(G[tau])**2)
    return a, G, eps
