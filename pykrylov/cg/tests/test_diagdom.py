# Test case for CG with a diagonally dominant matrix

import unittest
import sys

import numpy as np
from math import sqrt, sin, pi

from pykrylov.gallery import Poisson1dMatvec, Poisson2dMatvec
from pykrylov.linop import LinearOperator
from pykrylov.cg import CG
from pykrylov.tools import machine_epsilon


class Poisson1dTestCase(unittest.TestCase):

    def setUp(self):
        self.n = [10, 20, 100, 1000, 5000, 10000]
        self.eps = machine_epsilon()
        self.fmt = '%6d  %7d  %8.2e  %8.2e\n'
        hdrfmt = '%6s  %7s  %8s  %8s\n'
        hdr = hdrfmt % ('Size', 'Matvec', 'Resid', 'Error')
        sys.stderr.write('\n  Poisson1D tests\n')
        sys.stderr.write(hdr + '-' * len(hdr) + '\n')

    def tearDown(self):
        return

    def testPoisson1D(self):
        # Solve 1D Poisson systems of various sizes
        for n in self.n:

            lmbd_min = 4.0 * sin(pi/2.0/n) ** 2
            lmbd_max = 4.0 * sin((n-1)*pi/2.0/n) ** 2
            cond = lmbd_max/lmbd_min
            tol = cond * self.eps

            A = LinearOperator(n, n,
                               lambda x: Poisson1dMatvec(x),
                               symmetric=True)
            e = np.ones(n)
            rhs = A * e
            cg = CG(A, matvec_max=2*n, outputStream=sys.stderr)
            cg.solve(rhs)
            err = np.linalg.norm(e-cg.bestSolution)/sqrt(n)
            sys.stderr.write(self.fmt % (n, cg.nMatvec, cg.residNorm, err))
            self.assertTrue(np.allclose(e, cg.bestSolution, rtol=tol))


class Poisson2dTestCase(unittest.TestCase):

    def setUp(self):
        self.n = [10, 20, 100, 500]
        self.eps = machine_epsilon()
        self.fmt = '%6d  %7d  %8.2e  %8.2e\n'
        hdrfmt = '%6s  %7s  %8s  %8s\n'
        hdr = hdrfmt % ('Size', 'Matvec', 'Resid', 'Error')
        sys.stderr.write('\n  Poisson2D tests\n')
        sys.stderr.write(hdr + '-' * len(hdr) + '\n')

    def tearDown(self):
        return

    def testPoisson2D(self):
        # Solve 1D Poisson systems of various sizes
        for n in self.n:

            h = 1.0/n
            lmbd_min = 4.0/h/h*(sin(pi*h/2.0)**2 + sin(pi*h/2.0)**2)
            lmbd_max = 4.0/h/h*(sin((n-1)*pi*h/2.0)**2 + sin((n-1)*pi*h/2.0)**2)
            cond = lmbd_max/lmbd_min
            tol = cond * self.eps

            n2 = n*n
            A = LinearOperator(n2, n2,
                               lambda x: Poisson2dMatvec(x),
                               symmetric=True)
            e = np.ones(n2)
            rhs = A * e
            cg = CG(A, matvec_max=2*n2, outputStream=sys.stderr)
            cg.solve(rhs)
            err = np.linalg.norm(e-cg.bestSolution)/n
            sys.stderr.write(self.fmt % (n2, cg.nMatvec, cg.residNorm, err))

            # Adjust tol because allclose() uses infinity norm
            self.assertTrue(np.allclose(e, cg.bestSolution, rtol=err*n))


if __name__ == '__main__':
    unittest.main()
