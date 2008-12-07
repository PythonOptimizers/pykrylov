# Test case for CG with a diagonally dominant matrix

import unittest
import sys

import numpy as np
from math import sqrt, sin, pi
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

    def Poisson1dMatvec(self, x):
        # Matrix-vector product with a 1D Poisson matrix
        y = 2*x
        y[:-1] -= x[1:]
        y[1:] -= x[:-1]
        return y

    def testPoisson1D(self):
        # Solve 1D Poisson systems of various sizes
        for n in self.n:

            lmbd_min = 4.0 * sin(pi/2.0/n) ** 2
            lmbd_max = 4.0 * sin((n-1)*pi/2.0/n) ** 2
            cond = lmbd_max/lmbd_min
            tol = cond * self.eps

            e = np.ones(n)
            rhs = self.Poisson1dMatvec(e)
            cg = CG(self.Poisson1dMatvec,
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

    def Poisson2dMatvec(self, x):
        # Matrix-vector product with a 4D Poisson matrix
        n = int(sqrt(x.shape[0]))
        y = 4*x
        # Contribution of first block row
        y[:n-1] -= x[1:n]
        y[1:n] -= x[:n-1]
        y[:n] -= x[n:2*n]
        # Contribution of intermediate block rows
        for i in xrange(1,n-1):
            xi = x[i*n:(i+1)*n]   # This a view of x, not a copy
            yi = y[i*n:(i+1)*n]
            yi[:-1] -= xi[1:]
            yi[1:] -= xi[:-1]
            yi -= x[(i+1)*n:(i+2)*n]
            yi -= x[(i-1)*n:i*n]
        # Contribution of last block row
        y[(n-1)*n:] -= x[(n-2)*n:(n-1)*n]
        y[(n-1)*n+1:] -= x[(n-1)*n:-1]
        y[(n-1)*n:n*n-1] -= x[-n+1:]
        return y

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
            rhs = self.Poisson2dMatvec(e)
            cg = CG(self.Poisson2dMatvec,
                    matvec_max=2*n2,
                    outputStream=sys.stderr)
            cg.solve(rhs)
            err = np.linalg.norm(e-cg.bestSolution)/n
            sys.stderr.write(self.fmt % (n2, cg.nMatvec, cg.residNorm, err))

            # Adjust tol because allclose() uses infinity norm
            self.failUnless(np.allclose(e, cg.bestSolution, rtol=err*n))


if __name__ == '__main__':
    unittest.main()
