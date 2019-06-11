import unittest
import numpy as np
import scipy.linalg as linalg

from levinson import lev_durb, whittle_lev_durb
from clevinson import _lev_durb
from util import (block_toeplitz, system_rho,
                  is_stable)


class TestUtil(unittest.TestCase):
    rand_mat = lambda s: np.random.normal(size=(2, 2))

    def test_block_toep001(self):
        c = [-1, 2, 3, 8]
        Rb = block_toeplitz(c)
        Rs = linalg.toeplitz(c)
        np.testing.assert_array_equal(Rb, Rs)
        return

    def test_block_toep002(self):
        c = [-1, 2, 3, 8]
        r = [-1, -2.2, -3.8, -8.9]
        Rb = block_toeplitz(c, r)
        Rs = linalg.toeplitz(c, r)
        np.testing.assert_array_equal(Rb, Rs)
        return

    def test_block_toep003(self):
        C = [self.rand_mat(),
             self.rand_mat() + 1j * self.rand_mat()]
        T = block_toeplitz(C)

        np.testing.assert_array_equal(T[:2, :2], C[0])
        np.testing.assert_array_equal(T[2:, 2:], C[0])
        np.testing.assert_array_equal(T[2:, :2], C[1])
        np.testing.assert_array_equal(T[:2, 2:], C[1].T.conj())
        return

    def test_block_toep004(self):
        C = np.array([self.rand_mat(),
                      self.rand_mat()])
        R = np.array([C[0], self.rand_mat()])
        T = block_toeplitz(C, R)

        np.testing.assert_array_equal(T[:2, :2], C[0])
        np.testing.assert_array_equal(T[2:, 2:], C[0])
        np.testing.assert_array_equal(T[2:, :2], C[1])
        np.testing.assert_array_equal(T[:2, 2:], R[1])
        return

    def test_cov_seq001(self):
        """Univariate Covariance Sequences"""
        # Ensure we are actually producing PSD covariances
        p = 15
        try:
            for _ in range(1000):
                rand_cov_seq(p + 1, p, 1)
        except linalg.LinAlgError:
            self.fail("Non-PSD Matrix!")
        return

    def test_cov_seq002(self):
        """Multivariate Covariance Sequences"""
        n = 5
        p = 15
        try:
            for _ in range(500):
                rand_cov_seq(p * n + 1, p, n)
        except linalg.LinAlgError:
            self.fail("Non-PSD Matrix!")


class TestLevinsonDurbin(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        return

    def basic_test001(self):
        rx = np.array([1.0, 0.5, 0.5, 0.25])
        a_exp = np.array([1.0, -3. / 8, -3. / 8, 1. / 8])
        G_exp = np.array([-1. / 2, -1. / 3, 1. / 8])
        eps_exp = 21. / 32

        a, G, eps = lev_durb(rx)
        np.testing.assert_almost_equal(a, a_exp)
        np.testing.assert_almost_equal(G, G_exp)
        np.testing.assert_almost_equal(eps[-1], eps_exp)
        np.testing.assert_array_less(np.diff(eps), np.zeros(3))
        return

    def basic_test002(self):
        rx = np.array([2., -1., -0.25, 0.125])
        a_exp = np.array([1.0, 1.0, 7. / 8, 1. / 2])
        G_exp = np.array([0.5, 0.5, 0.5])
        eps_exp = 2 * (3. / 4)**3

        a, G, eps = lev_durb(rx)
        np.testing.assert_almost_equal(a, a_exp)
        np.testing.assert_almost_equal(G, G_exp)
        np.testing.assert_almost_equal(eps[-1], eps_exp)
        np.testing.assert_array_less(np.diff(eps), np.zeros(3))
        return

    def test_solve_toep001(self):
        """Small random system"""
        T = 12
        p = 4
        r = rand_cov_seq(T, p)
        R = linalg.toeplitz(r)

        u = np.zeros(p)
        u[0] = 1.0

        b_solve = linalg.solve(R, u)
        b_solve_toeplitz = linalg.solve_toeplitz(r, u)
        b_lev_durb, G, eps = lev_durb(r)
        b_lev_durb = b_lev_durb / eps[-1]

        np.testing.assert_almost_equal(
            b_solve, b_solve_toeplitz, decimal=7,
            err_msg="scipy.linalg solve and solve_toeplitz don't match!")

        np.testing.assert_almost_equal(
            b_solve_toeplitz, b_lev_durb, decimal=7)
        self.assertTrue(eps[-1] >= 0, "eps < 0!")
        np.testing.assert_array_less(np.abs(G), np.ones_like(G))
        np.testing.assert_almost_equal(1.0, np.sum(b_lev_durb * r))
        np.testing.assert_array_less(np.diff(eps), np.zeros(p - 1))
        return

    def test_solve_toep002(self):
        """Modest sized random system"""
        T = 1200
        p = 30
        r = rand_cov_seq(T, p)
        R = linalg.toeplitz(r)

        u = np.zeros(p)
        u[0] = 1.0

        b_solve = linalg.solve(R, u)
        b_solve_toeplitz = linalg.solve_toeplitz(r, u)
        b_lev_durb, G, eps = lev_durb(r)
        b_lev_durb = b_lev_durb / eps[-1]

        np.testing.assert_almost_equal(
            b_solve, b_solve_toeplitz, decimal=7,
            err_msg="scipy.linalg solve and solve_toeplitz don't match!")

        np.testing.assert_almost_equal(
            b_solve_toeplitz, b_lev_durb, decimal=7)
        self.assertTrue(eps[-1] >= 0, "eps < 0!")
        np.testing.assert_array_less(np.abs(G), np.ones_like(G))
        np.testing.assert_almost_equal(1.0, np.sum(b_lev_durb * r))
        np.testing.assert_array_less(np.diff(eps), np.zeros(p - 1))
        return

    def test_solve_toep003(self):
        """large random system"""
        # This isn't that big but my method to generate cov
        # sequences is really slow.
        T = 2000
        p = 500

        r = rand_cov_seq(T, p)
        R = linalg.toeplitz(r)

        u = np.zeros(p)
        u[0] = 1.0

        b_solve = linalg.solve(R, u)
        b_solve_toeplitz = linalg.solve_toeplitz(r, u)
        b_lev_durb, G, eps = lev_durb(r)
        b_lev_durb = b_lev_durb / eps[-1]

        np.testing.assert_almost_equal(
            b_solve, b_solve_toeplitz, decimal=7,
            err_msg="scipy.linalg solve and solve_toeplitz don't match!")

        np.testing.assert_almost_equal(
            b_solve_toeplitz, b_lev_durb, decimal=7)
        self.assertTrue(eps[-1] >= 0, "eps < 0!")
        np.testing.assert_array_less(np.abs(G), np.ones_like(G))
        np.testing.assert_almost_equal(1.0, np.sum(b_lev_durb * r))
        np.testing.assert_array_less(np.diff(eps), np.zeros(p - 1))
        return

    def test_solve_indefinite(self):
        # We can still solve systems with indefinite inputs
        T = p = 1000
        r = np.random.normal(size=T)
        R = linalg.toeplitz(r)

        u = np.zeros(p)
        u[0] = 1.0

        b_solve = linalg.solve(R, u)
        b_solve_toeplitz = linalg.solve_toeplitz(r, u)
        b_lev_durb, G, eps = lev_durb(r)
        b_lev_durb = b_lev_durb / eps[-1]

        np.testing.assert_almost_equal(
            b_solve, b_solve_toeplitz, decimal=7,
            err_msg="scipy.linalg solve and solve_toeplitz don't match!")

        np.testing.assert_almost_equal(
            b_solve_toeplitz, b_lev_durb, decimal=7)
        np.testing.assert_almost_equal(1.0, np.sum(b_lev_durb * r))
        return


class TestBlockLevinsonDurbin(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        return

    def _make_whittle_simple_test(self):
        T = 12
        p = 3
        n = 2
        r = rand_cov_seq(T, p, n)
        R = block_toeplitz(r)

        U = np.zeros((n * p, n))
        U[:n, :n] = np.eye(n)

        b_solve = linalg.solve(R, U)
        A, G, S = whittle_lev_durb(r)
        assert len(A) == p
        assert len(G) == p
        assert len(S) == p

        A_normed = [linalg.solve(S[-1], A_tau) for A_tau in A]
        A_normed = np.vstack(A_normed)
        B = [-A_tau for A_tau in A[1:]]
        return r, A, A_normed, b_solve, B, S

    def test_whittle_block001(self):
        """stability"""
        for _ in range(50):
            _, _, _, _, B, _ = self._make_whittle_simple_test()
            self.assertTrue(is_stable(B),
                            "System not stable!")
        return

    def test_whittle_block002(self):
        """solves YW"""
        for _ in range(50):
            r, A, _, _, _, _ = self._make_whittle_simple_test()
            assert_solves_yule_walker(A, r)
        return

    def test_whittle_block003(self):
        """Error PSD"""
        for _ in range(50):
            _, _, _, _, _, S = self._make_whittle_simple_test()
            self._assert_psd_sequence(S)
        return

    def test_whittle_block004(self):
        """Decreasing Error"""
        for _ in range(50):
            _, _, _, _, _, S = self._make_whittle_simple_test()
            self._assert_psd_decreasing_sequence(S)
        return

    def test_whittle_block005(self):
        """Toeplitz Solve"""
        _, _, A_normed, b_solve, _, _ = self._make_whittle_simple_test()
        np.testing.assert_almost_equal(A_normed, b_solve)
        return

    def _assert_psd_sequence(self, S):
        try:
            for tau in range(len(S)):
                linalg.cholesky(S[tau])
        except linalg.LinAlgError:
            self.fail("S[{}] is not positive definite!"
                      "".format(tau))
        return

    def _assert_psd_decreasing_sequence(self, S):
        try:
            for tau in range(len(S) - 1):
                linalg.cholesky(S[tau] - S[tau + 1])
        except linalg.LinAlgError:
            self.fail("S[{}] - S[{}] is not positive definite!"
                      "".format(tau, tau + 1))
        return


class TestcLevinsonDurbin(unittest.TestCase):
    def basic_test001(self):
        rx = np.array([1.0, 0.5, 0.5, 0.25])
        a_exp = np.array([1.0, -3. / 8, -3. / 8, 1. / 8])
        G_exp = np.array([-1. / 2, -1. / 3, 1. / 8])
        eps_exp = 21. / 32

        a, G, eps = _lev_durb(rx)
        np.testing.assert_almost_equal(a, a_exp)
        np.testing.assert_almost_equal(G, G_exp)
        np.testing.assert_almost_equal(eps, eps_exp)
        return


def rand_cov_seq(T, p, n=1):
    x = np.random.normal(size=(T, n))
    if n > 1:
        r = np.array(
            [np.sum([x[t][:, None] @ x[t - tau][:, None].T
                     for t in range(tau, T)], axis=0)
             for tau in range(p)]) / T
    else:
        r = np.array([np.sum([x[t] * x[t - tau]
                              for t in range(tau, T)])
                      for tau in range(p)]) / T

    R = block_toeplitz(r)
    try:
        linalg.cholesky(R)
    except linalg.LinAlgError:
        raise AssertionError("Sequence generated is not PSD!")
    return r


def assert_solves_yule_walker(A, R):
    """
    Check sum_{tau = 0}^p A[tau] @ R[s - tau] = 0 for each s = 1 to p
    """
    p = len(A)
    n = A.shape[1]
    for tau in range(1, p + 1):
        K_check = np.zeros((n, n))
        for s in range(p):
            print(tau - s - 1, s)
            if tau - s - 1 >= 0:
                K_check += A[s] @ R[tau - s - 1]
            else:
                K_check += A[s] @ R[s - tau + 1].T
        np.testing.assert_almost_equal(K_check, np.zeros((n, n)))
    return
