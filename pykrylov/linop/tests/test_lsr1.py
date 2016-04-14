"""Test LSR1 linear operators."""

from __future__ import division
import unittest
import numpy as np
from pykrylov.linop import lsr1
from pykrylov.tools import check_symmetric


class TestLSR1Operator(unittest.TestCase):
    """Test the various LSR1 linear operators."""

    def setUp(self):
        """Initialize."""
        self.n = 3
        self.npairs = 5
        self.B = lsr1.LSR1Operator(self.n, self.npairs)
        self.B_compact = lsr1.CompactLSR1Operator(self.n, self.npairs)
        self.H = lsr1.InverseLSR1Operator(self.n, self.npairs)

    def test_init(self):
        """Check that H = B = I initially."""
        assert self.B.insert == 0
        assert self.H.insert == 0
        assert np.allclose(self.B.full(), np.eye(self.n))
        assert np.allclose(self.B_compact.full(), np.eye(self.n))
        assert np.allclose(self.H.full(), np.eye(self.n))

    def test_structure(self):
        """Test that B and H are inverses of each other."""
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
        assert check_symmetric(self.B_compact)
        assert check_symmetric(self.H)

        C = self.B * self.H
        assert np.allclose(C.full(), np.eye(self.n))

        C_compact = self.B_compact * self.H
        assert np.allclose(C_compact.full(), np.eye(self.n))
