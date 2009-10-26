
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

    CG performs 1 matrix-vector product, 2 dot products and 3 daxpys per
    iteration.

    If a preconditioner is supplied, it needs to solve one preconditioning
    system per iteration. Our implementation is standard and follows [Kelley]_
    and [Templates]_.
    """

    def __init__(self, matvec, **kwargs):
        KrylovMethod.__init__(self, matvec, **kwargs)

        self.name = 'Conjugate Gradient'
        self.acronym = 'CG'
        self.prefix = self.acronym + ': '

        # Direction of nonconvexity if A is not positive definite
        self.infiniteDescent = None


    def solve(self, rhs, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the CG method.
        The vector `rhs` should be a Numpy array.

        :Keyword arguments and default values:

        +--------------+--------------------------------------+----+
        | `guess`      | Initial guess (Numpy array)          |  0 |
        +--------------+--------------------------------------+----+
        | `matvec_max` | Max. number of matrix-vector produts | 2n |
        +--------------+--------------------------------------+----+
        """
        n = rhs.shape[0]
        nMatvec = 0
        definite = True

        # Initial guess
        guess_supplied = 'guess' in kwargs.keys()
        x = kwargs.get('guess', np.zeros(n))
        matvec_max = kwargs.get('matvec_max', 2*n)

        # Initial residual vector
        r = -rhs
        if guess_supplied:
            r += self.matvec(x)
            nMatvec += 1

        # Initial preconditioned residual vector
        if self.precon is not None:
            y = self.precon(r)
        else:
            y = r

        ry = np.dot(r,y)
        self.residNorm0 = residNorm = sqrt(ry)
        threshold = max( self.abstol, self.reltol * self.residNorm0 )

        p = -r   # Initial search direction (copy not to overwrite rhs if x=0)
        
        while residNorm > threshold and nMatvec < matvec_max and definite:

            Ap  = self.matvec(p)
            nMatvec += 1
            pAp = np.dot(p, Ap)
            
            #if pAp <= 0:
            #    self._write('Coefficient matrix is not positive definite\n')
            #    self.infiniteDescent = p
            #    definite = False
            #    continue

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
