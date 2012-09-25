# Test case for CG with a diagonally dominant matrix

import unittest
import sys

import numpy as np
from math import sqrt, sin, pi

from pykrylov.gallery import Poisson1dMatvec, Poisson2dMatvec
from pykrylov.cg import CG

def macheps():
    "compute machine epsilon"
    eps = 1.0
    while (1.0 + eps > 1.0):
        eps /= 2.0
    return 2.0 * eps


class Poisson1dTestCase(unittest.TestCase):

    def setUp(self):
        self.n = [10, 20, 100, 1000, 5000, 10000]
        self.eps = macheps()
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

            e = np.ones(n)
            rhs = Poisson1dMatvec(e)
            cg = CG(Poisson1dMatvec,
                    matvec_max=2*n,
                    outputStream=sys.stderr)
            cg.solve(rhs)
            err = np.linalg.norm(e-cg.bestSolution)/sqrt(n)
            sys.stderr.write(self.fmt % (n, cg.nMatvec, cg.residNorm, err))
            self.failUnless(np.allclose(e, cg.bestSolution, rtol=tol))


class Poisson2dTestCase(unittest.TestCase):

    def setUp(self):
        self.n = [10, 20, 100, 500]
        self.eps = macheps()
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
            e = np.ones(n2)
            rhs = Poisson2dMatvec(e)
            cg = CG(Poisson2dMatvec,
                    matvec_max=2*n2,
                    outputStream=sys.stderr)
            cg.solve(rhs)
            err = np.linalg.norm(e-cg.bestSolution)/n
            sys.stderr.write(self.fmt % (n2, cg.nMatvec, cg.residNorm, err))

            # Adjust tol because allclose() uses infinity norm
            self.failUnless(np.allclose(e, cg.bestSolution, rtol=err*n))


if __name__ == '__main__':
    unittest.main()
