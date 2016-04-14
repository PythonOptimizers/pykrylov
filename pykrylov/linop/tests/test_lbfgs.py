"""Test LBFGS linear operators."""

from __future__ import division
import unittest
import numpy as np
from pykrylov.linop import lbfgs
from pykrylov.tools import check_symmetric, check_positive_definite


class TestLBFGSOperator(unittest.TestCase):
    """Test the various LBFGS linear operators."""

    def setUp(self):
        """Initialize."""
        self.n = 10
        self.npairs = 5
        self.B = lbfgs.LBFGSOperator(self.n, self.npairs)
        self.B_compact = lbfgs.CompactLBFGSOperator(self.n, self.npairs)
        self.H = lbfgs.InverseLBFGSOperator(self.n, self.npairs)

    def test_init(self):
        """Check that H = B = I initially."""
        assert self.B.insert == 0
        assert self.H.insert == 0
        assert np.allclose(self.B.full(), np.eye(self.n))
        assert np.allclose(self.H.full(), np.eye(self.n))

    def test_negative_curvature(self):
        """Test that negative curvature isn't captured."""
        s = np.random.random(self.n)
        z = np.zeros(self.n)
        self.B.store(s, -s)
        self.B.store(s, z)
        assert self.B.insert == 0
        self.H.store(s, -s)
        self.H.store(s, z)
        assert self.H.insert == 0

    def test_structure(self):
        """Test that B and H are spd and inverses of each other."""
        # Insert a few {s,y} pairs.
        for _ in range(self.npairs + 2):
            s = np.random.random(self.n)
            y = np.random.random(self.n)
            self.B.store(s, y)
            self.B_compact.store(s, y)
            self.H.store(s, y)

        assert self.B.insert == 2
        assert self.H.insert == 2

        assert check_symmetric(self.B)
        assert check_symmetric(self.H)
        assert check_positive_definite(self.B)
        assert check_positive_definite(self.H)

        C = self.B * self.H
        assert np.allclose(C.full(), np.eye(self.n))
        C_compact = self.B_compact * self.H
        assert np.allclose(C_compact.full(), np.eye(self.n))
