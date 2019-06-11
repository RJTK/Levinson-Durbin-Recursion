import unittest
import numpy as np
import scipy.linalg as linalg

from levinson import lev_durb


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
        np.testing.assert_almost_equal(eps, eps_exp)
        return

    def basic_test002(self):
        rx = np.array([2., -1., -0.25, 0.125])
        a_exp = np.array([1.0, 1.0, 7. / 8, 1. / 2])
        G_exp = np.array([0.5, 0.5, 0.5])
        eps_exp = 2 * (3. / 4)**3

        a, G, eps = lev_durb(rx)
        np.testing.assert_almost_equal(a, a_exp)
        np.testing.assert_almost_equal(G, G_exp)
        np.testing.assert_almost_equal(eps, eps_exp)
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
        b_lev_durb = b_lev_durb / eps

        np.testing.assert_almost_equal(
            b_solve, b_solve_toeplitz, decimal=7,
            err_msg="scipy.linalg solve and solve_toeplitz don't match!")

        np.testing.assert_almost_equal(
            b_solve_toeplitz, b_lev_durb, decimal=7)
        self.assertTrue(eps >= 0, "eps < 0!")
        np.testing.assert_array_less(np.abs(G), np.ones_like(G))
        np.testing.assert_almost_equal(1.0, np.sum(b_lev_durb * r))
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
        b_lev_durb = b_lev_durb / eps

        np.testing.assert_almost_equal(
            b_solve, b_solve_toeplitz, decimal=7,
            err_msg="scipy.linalg solve and solve_toeplitz don't match!")

        np.testing.assert_almost_equal(
            b_solve_toeplitz, b_lev_durb, decimal=7)
        self.assertTrue(eps >= 0, "eps < 0!")
        np.testing.assert_array_less(np.abs(G), np.ones_like(G))
        np.testing.assert_almost_equal(1.0, np.sum(b_lev_durb * r))
        return

    def test_solve_toep003(self):
        """large random system"""
        T = 5000
        p = 1000

        r = rand_cov_seq(T, p)
        R = linalg.toeplitz(r)

        u = np.zeros(p)
        u[0] = 1.0

        b_solve = linalg.solve(R, u)
        b_solve_toeplitz = linalg.solve_toeplitz(r, u)
        b_lev_durb, G, eps = lev_durb(r)
        b_lev_durb = b_lev_durb / eps

        np.testing.assert_almost_equal(
            b_solve, b_solve_toeplitz, decimal=7,
            err_msg="scipy.linalg solve and solve_toeplitz don't match!")

        np.testing.assert_almost_equal(
            b_solve_toeplitz, b_lev_durb, decimal=7)
        self.assertTrue(eps >= 0, "eps < 0!")
        np.testing.assert_array_less(np.abs(G), np.ones_like(G))
        np.testing.assert_almost_equal(1.0, np.sum(b_lev_durb * r))
        return


def rand_cov_seq(T, p):
    x = np.random.normal(size=T)
    r = np.array(
        [np.sum([x[t] * x[t - tau] for t in range(p, T)])
         for tau in range(p)])
    R = linalg.toeplitz(r)
    try:
        linalg.cholesky(R)
    except linalg.LinAlgError:
        raise AssertionError("Sequence generated is not PSD!")
    return r
