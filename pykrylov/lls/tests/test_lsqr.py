"""Test LBFGS linear operators."""

from __future__ import division
import unittest
import numpy as np
from pykrylov.lls.lsqr import LSQRFramework
from pykrylov.linop import linop_from_ndarray


class TestLSQR(unittest.TestCase):
    """Test LSQR method on small instances."""

    def setUp(self):
        """Initialize."""
        self.n = 10
        self.npairs = 5
        A = np.array([[1, 0], [1, 1], [1, 3], [1, 4]], dtype=np.double)
        self.A = linop_from_ndarray(A, symmetric=False)
        self.rhs = np.array([0, 8, 8, 20], dtype=np.double)

    def test_trust_region_constraint(self):
        """Check solution using a trust-region constraint.

        Solve the linear system
        [ 1  0 ] [x] = [0 ]
        [ 1  1 ] [y]   [8 ]
        [ 1  3 ]       [8 ]
        [ 1  4 ]       [20]

        for multiple trust-region constraints.
        """
        lsqr = LSQRFramework(self.A)

        # trust-region constraint is not active
        lsqr.solve(self.rhs, radius=5, show=True)
        np.allclose(lsqr.x, np.array([1., 4.]))

        # trust-region constraint is active
        lsqr.solve(self.rhs, radius=np.sqrt(17), show=True)
        np.allclose(lsqr.x, np.array([1., 4.]))

        # trust-region constraint is active
        lsqr.solve(self.rhs, radius=50. / 13, show=True)
        np.allclose(lsqr.x, np.array([14. / 13, 48. / 13]))
