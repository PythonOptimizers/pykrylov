
__docformat__ = 'restructuredtext'

import numpy as np
from math import sqrt

from pykrylov.generic import KrylovMethod

class CGS( KrylovMethod ):
    """
    A pure Python implementation of the conjugate gradient squared (CGS)
    algorithm. CGS may be used to solve unsymmetric systems of linear equations,
    i.e., systems of the form

        A x = b

    where the matrix A may be unsymmetric.

    CGS requires 2 matrix-vector products with A, 3 dot products and 7 daxpys
    per iteration. It does not require products with the transpose of A.

    If a preconditioner is supplied, CGS needs to solve two preconditioning
    systems per iteration.
    """

    def __init__(self, matvec, **kwargs):
        KrylovMethod.__init__(self, matvec, **kwargs)

        self.prefix = 'CGS: '
        self.name = 'Conjugate Gradient Squared'

    def solve(self, rhs, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the CGS method.
        The vector `rhs` should be a Numpy array. An optional argument `guess`
        may be supplied, with an initial guess as a Numpy array. By default,
        the initial guess is the vector of zeros.
        """
        n = rhs.shape[0]
        nMatvec = 0

        # Initial guess is zero unless one is supplied
        guess_supplied = 'guess' in kwargs.keys()
        x = kwargs.get('guess', np.zeros(n))

        r0 = rhs  # Fixed vector throughout
        if guess_supplied:
            r0 = rhs - self.matvec(x)

        rho = np.dot(r0,r0)
        residNorm = sqrt(rho)
        self.residNorm0 = residNorm
        threshold = max( self.abstol, self.reltol * self.residNorm0 )
        if self.verbose:
            self._write('Initial residual = %8.2e\n' % self.residNorm0)
            self._write('Threshold = %8.2e\n' % threshold)

        finished = (residNorm <= threshold or nMatvec >= self.matvec_max)

        if not finished:
            r = r0.copy()   # Initial residual vector
            u = r0
            p = r0.copy()

        while not finished:

            if self.precon is not None:
                y = self.precon(p)
            else:
                y = p

            v = self.matvec(y) ; nMatvec += 1
            sigma = np.dot(r0,v)
            alpha = rho/sigma
            q = u - alpha * v
            
            if self.precon is not None:
                z = self.precon(u+q)
            else:
                z = u+q

            # Update solution and residual
            x += alpha * z
            Az = self.matvec(z) ; nMatvec += 1
            r -= alpha * Az

            # Update residual norm and check convergence
            residNorm = sqrt(np.dot(r,r))

            if residNorm <= threshold or nMatvec >= self.matvec_max:
                finished = True
                continue

            rho_next = np.dot(r0,r)
            beta = rho_next/rho
            rho = rho_next
            u = r + beta * q
            
            # Update p in-place
            p *= beta
            p += q
            p *= beta
            p += u

            # Display current info if requested
            if self.verbose:
                self._write('%5d  %8.2e\n' % (nMatvec, residNorm))


        self.nMatvec = nMatvec
        self.bestSolution = x
        self.residNorm = residNorm
