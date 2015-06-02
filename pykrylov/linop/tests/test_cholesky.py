"""Test Cholesky linear operator. """

from __future__ import division
import unittest
import numpy as np
import scipy as sp
from pykrylov.tools import check_symmetric, check_positive_definite

try:
    from pykrylov.linop import CholeskyOperator

    class TestCholeskyOperator(unittest.TestCase):
        """Test the Cholesky linear operator."""

        def setUp(self):
            """Initialize."""
            self.n = 10
            _A = np.random.random((self.n, self.n))
            A = np.dot(_A.T, _A)
            self.A = sp.sparse.construct.csc_matrix(A)
            self.M = CholeskyOperator(self.A)

        def test_symmetric(self):
            """Check that the factorization is a symmetric operator."""
            assert check_symmetric(self.M)

        def test_positive_definite(self):
            """Check that the factorization is a positive-definite operator."""
            assert check_positive_definite(self.M)

        def test_solve_system(self):
            """Check solution of linear system."""
            e = np.ones(self.n)
            b = self.A * e
            assert np.allclose(e, self.M * b)

except:
    pass
