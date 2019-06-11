"""
Implementation of Levinson Recursion and other associated routines,
particularly the Block Toeplitz versions of Whittle and Akaike.
"""

import numpy as np
import numba


# TODO: It would be useful to "break-out" the inner portion of
# TODO: the loop so that we can obtain all the solutions of order
# TODO: up to p = len(r) - 1.  This would be highly desirable for
# TODO: efficient model order selection.
@numba.jit(nopython=True, cache=False)
def lev_durb(r):
    """
    Comsumes a length p + 1 vector r = [r(0), ..., r(p)] and returns
    (a, G, eps) as follows:

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

    One of the key advantages of this algorithm is that the resulting
    filter is guaranteed to be stable, and the prediction error is directly
    available as a byproduct.

    Moreover, the sequence G has the property forall tau: |G(tau)| < 1.0
    if and only if r is a positive definite covariance sequence.

    reference:

    @book{hayes2009statistical,
      title={Statistical digital signal processing and modeling},
      author={Hayes, Monson H},
      year={2009},
      publisher={John Wiley \& Sons}
    }
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


@numba.jit(nopython=True, cache=True)
def whittle_lev_durb(R):
    """
    Comsumes a length p + 1 vector R = [R(0), ..., R(p)] of n x n
    block matrices which must be a valid (vector-)autocovariance sequence
    (i.e. the block-toeplitz matrix formed from R must be positive
    semi-definite) and returns (A, G, S) as follows:

    returns:
      - A (List[np.array]): Length p + 1 array (with a[0] = np.eye(n))
        consisting of the filter coefficients for an all-pole model of a
        signal having autocovariance R(tau).
      - G (List[np.array]): Length p list of reflection coefficient matrices.
      - S (np.array): The variance matrix achieved by the all-pole
        model.  S is guaranteed to be positive semi-definite

    We are returning a solution to: block-toep(R) @ A = e1 (x) S where (x)
    denote kronecker product and e1 is the first canonical basis vector.
    The (matrix-)variables are A[1:] and S.

    Fortunately, the block version of this algorithm also enjoys the
    stability property of the scalar version, i.e. det |A(z)| has it's
    zeros within the unit circle.
    """
    p = len(R) - 1
    n = R[0].shape[0]
    A = [np.zeros((n, n)) for tau in range(p + 1)]  # Forward coeffs
    A_bar = np.copy(A)  # Backward coeffs

    V = np.zeros((n, n))  # Forward error variance
    V_bar = np.copy(V)  # Backward error variance

    V[0] = R[0]
    V_bar[0] = R[0]

    for tau in range(p):
        pass
    return
