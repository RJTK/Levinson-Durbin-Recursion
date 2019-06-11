"""
Cython implementation of levinson-durbin recursion.  This is
dramatically slower than the numba version, probably because
I don't really know how to use cython.
"""

cimport cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def _lev_durb(r):
    cdef long p = len(r) - 1
    cdef np.ndarray[double, ndim=1] a = np.zeros(p + 1)
    cdef np.ndarray[double, ndim=1] a_cpy = np.zeros(p + 1)
    cdef np.ndarray[double, ndim=1] G = np.zeros(p)
    cdef double eps = r[0]
    cdef double conv
    cdef long tau, s

    a[0] = 1.0

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
    
