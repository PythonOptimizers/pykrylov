
__docformat__ = 'restructuredtext'

import numpy as np
from math import sqrt

from pykrylov.generic import KrylovMethod

class CG( KrylovMethod ):
    """
    A pure Python implementation of the conjugate gradient (CG) algorithm. The
    conjugate gradient algorithm may be used to solve symmetric positive
    definite systems of linear equations, i.e., systems of the form

        A x = b

    where the matrix A is square, symmetric and positive definite. This is
    equivalent to solving the unconstrained convex quadratic optimization
    problem

        minimize    -<b,x> + 1/2 <x, Ax>

    in the variable x.
    """

    def __init__(self, matvec, **kwargs):
        KrylovMethod.__init__(self, matvec, **kwargs)

        self.prefix = 'CG: '
        self.name = 'Conjugate Gradient'

        # Direction of nonconvexity if A is not positive definite
        self.infiniteDescent = None


    def solve(self, rhs, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side. The vector rhs`
        should be a Numpy array.
        """
        n = rhs.shape[0]
        nMatvec = 0
        definite = True

        x = np.zeros(n)
        r = -rhs  # Initial residual vector

        if self.precon is not None:
            y = self.precon(r)
        else:
            y = r     # Initial preconditioned residual vector

        ry = np.dot(r,y)
        self.residNorm0 = residNorm = sqrt(ry)
        threshold = max( self.abstol, self.reltol * self.residNorm0 )

        p = -r   # Initial search direction (copy so as not to overwrite rhs)
        
        while residNorm > threshold and nMatvec < self.matvec_max and definite:

            Ap  = self.matvec(p)
            nMatvec += 1
            pAp = np.dot(p, Ap)
            
            if pAp <= 0:
                self._write('Coefficient matrix is not positive definite\n')
                self.infiniteDescent = p
                definite = False
                continue

            # Compute step length
            alpha = ry/pAp

            # Update estimate and residual
            x += alpha * p
            r += alpha * Ap
            
            # Compute preconditioned residual
            if self.precon is not None:
                y = self.precon(r)
            else:
                y = r

            # Update preconditioned residual norm
            ry_next = np.dot(r,y)

            # Update search direction
            beta = ry_next/ry
            p *= beta
            p -= r

            ry = ry_next
            residNorm = sqrt(ry)


        self.nMatvec = nMatvec
        self.bestSolution = x
        self.residNorm = residNorm
