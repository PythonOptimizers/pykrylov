
__docformat__ = 'restructuredtext'

import numpy as np
from math import sqrt

from pykrylov.generic import KrylovMethod

class TFQMR( KrylovMethod ):
    """
    A pure Python implementation of the transpose-free quasi-minimum residual
    (TFQMR) algorithm. TFQMR may be used to solve unsymmetric systems of linear
    equations, i.e., systems of the form

        `A x = b`

    where the matrix `A` may be unsymmetric.

    TFQMR requires 2 matrix-vector products with `A`, 4 dot products and
    10 daxpys per iteration. It does not require products with the transpose
    of `A`.

    If a preconditioner is supplied, TFQMR needs to solve 2 preconditioning
    systems per iteration.
    """

    def __init__(self, matvec, **kwargs):
        KrylovMethod.__init__(self, matvec, **kwargs)

        self.name = 'Transpose-Free Quasi-Minimum Residual'
        self.acronym = 'TFQMR'
        self.prefix = self.acronym + ': '

    def solve(self, rhs, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the TFQMR method.
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
            y = r0.copy()   # Initial residual vector
            w = r0.copy()
            d = np.zeros(n)
            theta = 0.0
            eta = 0.0
            k = 0
            if self.precon is not None:
                z = self.precon(y)
            else:
                z = y

            u = self.matvec(z) ; nMatvec += 1
            v = u.copy()

        while not finished:

            k += 1
            sigma = np.dot(r0,v)
            alpha = rho/sigma
            
            # First pass
            w -= alpha * u
            d *= theta * theta * eta / alpha
            d += z
            theta = np.linalg.norm(w)/residNorm
            c = 1.0/sqrt(1 + theta*theta)
            residNorm *= theta * c
            eta = c * c * alpha
            x += eta * d
            m = 2.0 * k - 1.0
            if residNorm * sqrt(m+1) < threshold or nMatvec >= self.matvec_max:
                finished = True
                continue

            # Second pass
            m += 1
            y -= alpha * v

            if self.precon is not None:
                z = self.precon(y)
            else:
                z = y

            u = self.matvec(z) ; nMatvec += 1
            w -= alpha * u
            d *= theta * theta * eta / alpha
            d += z
            theta = np.linalg.norm(w)/residNorm
            c = 1.0/sqrt(1 + theta*theta)
            residNorm *= theta * c
            eta = c * c * alpha
            x += eta * d
            if residNorm * sqrt(m+1) < threshold or nMatvec >= self.matvec_max:
                finished = True
                continue

            # Final updates
            rho_next = np.dot(r0,w)
            beta = rho_next/rho
            rho = rho_next

            # Update y
            y *= beta
            y += w

            # Partial update of v with current u
            v *= beta
            v += u
            v *= beta

            # Update u
            if self.precon is not None:
                z = self.precon(y)
            else:
                z = y

            u = self.matvec(z) ; nMatvec += 1

            # Complete update of v
            v += u

            # Display current info if requested
            if self.verbose:
                self._write('%5d  %8.2e\n' % (nMatvec, residNorm))


        self.nMatvec = nMatvec
        self.bestSolution = x
        self.residNorm = residNorm
