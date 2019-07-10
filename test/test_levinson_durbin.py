import unittest
import numpy as np
import numpy.linalg as linalg

from scipy.linalg import toeplitz, solve_toeplitz

from levinson.levinson import (lev_durb, whittle_lev_durb,
                               yule_walker, _whittle_lev_durb,
                               reflection_coefs, step_up,
                               A_to_B, fit_model_ret_plac,
                               system_rho, is_stable, B_to_A)
try:
    from .util import (block_toeplitz)
except ModuleNotFoundError:  # When debugging interactively
    from util import (block_toeplitz)


class TestUtil(unittest.TestCase):
    rand_mat = lambda: np.random.normal(size=(2, 2))

    def test_block_toep001(self):
        c = [-1, 2, 3, 8]
        Rb = block_toeplitz(c)
        Rs = toeplitz(c)
        np.testing.assert_array_equal(Rb, Rs)
        return

    def test_block_toep002(self):
        c = [-1, 2, 3, 8]
        r = [-1, -2.2, -3.8, -8.9]
        Rb = block_toeplitz(c, r)
        Rs = toeplitz(c, r)
        np.testing.assert_array_equal(Rb, Rs)
        return

    def test_block_toep003(self):
        C = [TestUtil.rand_mat(),
             TestUtil.rand_mat() + 1j * TestUtil.rand_mat()]
        T = block_toeplitz(C)

        np.testing.assert_array_equal(T[:2, :2], C[0])
        np.testing.assert_array_equal(T[2:, 2:], C[0])
        np.testing.assert_array_equal(T[2:, :2], C[1])
        np.testing.assert_array_equal(T[:2, 2:], C[1].T.conj())
        return

    def test_block_toep004(self):
        C = np.array([TestUtil.rand_mat(),
                      TestUtil.rand_mat()])
        R = np.array([C[0], TestUtil.rand_mat()])
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
        for _ in range(1000):
            r = rand_cov_seq(p + 1, p, 1)
            self.assertTrue(is_cov_sequence(r), "Non-PSD Sequence!")
        return

    def test_cov_seq002(self):
        """Multivariate Covariance Sequences"""
        n = 5
        p = 15
        for _ in range(500):
            r = rand_cov_seq(p * n + 1, p, n)
            self.assertTrue(is_cov_sequence(r), "Non-PSD Sequence!")
        return

    def test_is_cov_sequence(self):
        r = np.random.normal(size=20)
        self.assertFalse(is_cov_sequence(r))

        r = np.random.normal(size=(20, 5, 5))
        self.assertFalse(is_cov_sequence(r))
        return


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
        R = toeplitz(r)

        u = np.zeros(p)
        u[0] = 1.0

        b_solve = linalg.solve(R, u)
        b_solve_toeplitz = solve_toeplitz(r, u)
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
        R = toeplitz(r)

        u = np.zeros(p)
        u[0] = 1.0

        b_solve = linalg.solve(R, u)
        b_solve_toeplitz = solve_toeplitz(r, u)
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
        R = toeplitz(r)

        u = np.zeros(p)
        u[0] = 1.0

        b_solve = linalg.solve(R, u)
        b_solve_toeplitz = solve_toeplitz(r, u)
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
        R = toeplitz(r)

        u = np.zeros(p)
        u[0] = 1.0

        b_solve = linalg.solve(R, u)
        b_solve_toeplitz = solve_toeplitz(r, u)
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
        T = 120
        p = 8
        n = 10
        r = rand_cov_seq(T, p, n)
        R = block_toeplitz([r[k].T for k in range(p)])

        U = np.zeros((n * p, n))
        U[:n, :n] = np.eye(n)

        b_solve = linalg.solve(R, U)
        A, G, S = whittle_lev_durb(r)
        assert len(A) == p
        assert len(S) == p

        A_normed = [A_tau.T @ linalg.inv(S[-1]) for A_tau in A]
        A_normed = np.vstack(A_normed)
        B = A_to_B(A)
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
            r, A, _, _, _, S = self._make_whittle_simple_test()
            assert_solves_yule_walker(A, r)
            YW = yule_walker(A, r)
            np.testing.assert_almost_equal(YW[0], S[-1])
        return

    def test_whittle_block003(self):
        """Error PSD"""
        for _ in range(50):
            _, _, _, _, _, S = self._make_whittle_simple_test()
            self._assert_all_psd(S)
        return

    def test_whittle_block004(self):
        """Decreasing Error"""
        for _ in range(50):
            _, _, _, _, _, S = self._make_whittle_simple_test()
            self._assert_psd_decreasing_sequence(S)
        return

    def test_whittle_block005(self):
        """Toeplitz Solve"""
        for _ in range(20):
            _, _, A_normed, b_solve, _, _ = self._make_whittle_simple_test()
            np.testing.assert_almost_equal(A_normed, b_solve)
        return


    def test_whittle_block006(self):
        """Check PLAC is COV sequence -- is it???"""
        # TODO: Is this actually even true?
        # TODO: Go back and look at math, I might have just guessed
        # TODO: this should hold without checking...
        T = 120
        p = 8
        n = 10
        for _ in range(20):
            r = rand_cov_seq(T, p, n)
            _, P = fit_model_ret_plac(r)
            self.assertTrue(is_cov_sequence(P))
        return

    def test_A_to_B(self):
        for _ in range(20):
            r, A, _, _, _, S = self._make_whittle_simple_test()
            B = A_to_B(A)
            A_cycled = B_to_A(B)
            np.testing.assert_almost_equal(A, A_cycled)
        return

    def _assert_all_psd(self, S):
        try:
            for tau in range(len(S)):
                linalg.cholesky(S[tau])
        except linalg.LinAlgError:
            self.fail("S[{}] is not positive definite!"
                      "".format(tau))
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


class TestStepUp(unittest.TestCase):
    def SetUp(self):
        np.random.seed(0)
        return

    def _make_refl_coefs(self):
        def get_mat(n):
            X = np.random.normal(size=(n, n + 1))
            Y = np.random.normal(size=(n, n + 1))
            return X @ Y.T

        def get_cov(n):
            X = np.random.normal(size=(n, n + 1))
            return X @ X.T

        def _max_sv(R):
            return np.max(linalg.svd(R, compute_uv=False))

        def _min_sv(R):
            return np.min(linalg.svd(R, compute_uv=False))

        p = 3
        n = 3
        d = 0.9
        Delta = np.stack([get_mat(n)
                          for tau in range(p + 1)], 0)
        Delta[0] = np.eye(n)

        det_max = np.max(np.abs(linalg.det(Delta)))
        Delta[1:] = Delta[1:] * (d / det_max)**(1. / n)
        Delta_bar = np.moveaxis(Delta, 1, 2)

        # ev_max = np.max(np.abs(linalg.eigvals(Delta)))
        # Delta[1:] = Delta[1:] * (d / ev_max)
        # Delta_bar = np.moveaxis(Delta, 1, 2)

        V = np.zeros((p + 1, n, n))
        V[0] = get_cov(n)
        for tau in range(p):
            R = get_cov(n)
            R = 0.9 * _min_sv(V[tau]) * R / _max_sv(R)
            V[tau + 1] = V[tau] - R

        G, G_bar = reflection_coefs(Delta, Delta_bar, V,
                                    np.moveaxis(V, 1, 2))
        return G, G_bar

    def _make_data(self):
        T = 12
        p = 5
        n = 2
        r = rand_cov_seq(T, p, n)
        return r

    def test001(self):
        # Checks some properties of G and delta
        for _ in range(100):
            r = self._make_data()
            _, _, Delta, Delta_bar, V, V_bar = _whittle_lev_durb(r)
            assert_valid_partial_refl_coefs(Delta, Delta_bar)

            G, G_bar = reflection_coefs(Delta, Delta_bar, V, V_bar)
            assert_valid_refl_coefs(G, G_bar)
        return

    def test002(self):
        for _ in range(20):
            r = self._make_data()
            A, A_bar, Delta, Delta_bar, V, V_bar = _whittle_lev_durb(r)
            G, G_bar = reflection_coefs(Delta, Delta_bar, V, V_bar)
            A_su, A_bar_su = step_up(G, G_bar)
            np.testing.assert_almost_equal(A, A_su)
            np.testing.assert_almost_equal(A_bar, A_bar_su)
        return

    @unittest.skip("Don't need to use this")
    def test003(self):
        for _ in range(20):
            G, G_bar = self._make_refl_coefs()
            assert_valid_refl_coefs(G, G_bar)
        return

    @unittest.skip("Need not hold")
    def test004(self):
        G, G_bar = self._make_refl_coefs()
        A, A_bar = step_up(G, G_bar)
        B, B_bar = A_to_B(A), A_to_B(A_bar)
        self.assertTrue(is_stable(B))
        self.assertTrue(is_stable(B_bar))
        return


class TestcLevinsonDurbin(unittest.TestCase):
    @unittest.skip("Not using Cython")
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
    return r


def is_cov_sequence(r):
    R = block_toeplitz(r)
    try:
        linalg.cholesky(R)
    except linalg.LinAlgError:
        return False
    return True


def assert_solves_yule_walker(A, r):
    p = len(A) - 1
    n = A.shape[1]

    YW = yule_walker(A, r)
    for k in range(1, p + 1):
        np.testing.assert_almost_equal(YW[k], np.zeros((n, n)))
    return


def assert_valid_refl_coefs(G, G_bar):
    np.testing.assert_array_almost_equal(linalg.det(G),
                                         linalg.det(G_bar))
    return

def assert_valid_partial_refl_coefs(Delta, Delta_bar):
    np.testing.assert_array_almost_equal(
        Delta, np.moveaxis(Delta_bar, 1, 2))

    np.testing.assert_array_less(linalg.det(Delta)[1:],
                                 np.ones(len(Delta) - 1))
    EV = linalg.eigvals(Delta)[1:]
    EV_bar = linalg.eigvals(Delta_bar)[1:]
    np.testing.assert_array_almost_equal(EV, EV_bar)
    # np.testing.assert_array_less(np.abs(EV), np.ones_like(EV))
    # np.testing.assert_array_less(np.abs(EV_bar), np.ones_like(EV_bar))
    return
