"""
Implementation of Levinson Recursion and other associated routines,
particularly the Block Toeplitz versions of Whittle and Akaike.
"""

import numpy as np
import numba


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
      - eps (np.array): The sequence of errors achieved by all-pole
        models of progressively larger order.  eps is guaranteed to
        satisfy eps >= 0.

    NOTE: We don't handle complex data

    We get a solution to the system R @ a = eps * e1 where R = toep(r)
    and e1 is the first canonical basis vector.  The variables are a[1:]
    and eps.  NOTE: For epsilon we are returning a sequence of errors for
    progressively larger systems.

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
    eps = np.zeros(p + 1)
    eps[0] = r[0]

    for tau in range(p):
        # Compute reflection coefficient
        conv = r[tau + 1]
        for s in range(1, tau + 1):
            conv = conv + a[s] * r[tau - s + 1]
        G[tau] = -conv / eps[tau]

        # Update 'a' vector
        a_cpy = np.copy(a)
        for s in range(1, tau + 1):
            a_cpy[s] = a[s] + G[tau] * np.conj(a[tau - s + 1])
        a = a_cpy
        a[tau + 1] = G[tau]
        eps[tau + 1] = eps[tau] * (1 - np.abs(G[tau])**2)
    return a, G, eps


# @numba.jit(nopython=True, cache=True)
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

    A = np.zeros((p + 1, n, n))
    A_bar = np.copy(A)  # Backward coeffs

    A[0] = np.eye(n)
    A_bar[0] = np.eye(n)

    V = np.zeros((p + 1, n, n))  # Forward error variance
    Delta = np.zeros((p + 1, n, n))  # Reflection coefficients

    for tau in range(p):
        Delta[tau] = np.zeros((n, n))  # Forward reflection coefficients
        Delta_bar = np.zeros((n, n))  # Backward reflection coefficients
        V_bar = np.zeros((n, n))  # Backward error variance

        for s in range(tau + 1):
            V[tau] = V[tau] + A[s] @ R[s].T
            Delta[tau] = Delta[tau] + A[s] @ R[tau - s + 1]

            V_bar = V_bar + A_bar[s] @ R[s]
            Delta_bar = Delta_bar + A_bar[s] @ R[tau - s + 1].T

        A_cpy = np.copy(A)
        A_bar_cpy = np.copy(A_bar)

        # TODO: Use solve and the Hermitian PSD property of V
        A_cpy[tau + 1] = -Delta[tau] @ np.linalg.inv(V_bar)
        A_bar_cpy[tau + 1] = -Delta_bar[tau] @ np.linalg.inv(V[tau])

        for s in range(1, tau + 1):
            A_cpy[s] = A[s] + A_cpy[tau + 1] @ A_bar[tau - s + 1]
            A_bar_cpy[s] = A_bar[s] + A_bar_cpy[tau + 1] @ A[tau - s + 1]

        A = np.copy(A_cpy)
        A_bar = np.copy(A_bar_cpy)
    return A, Delta, V
