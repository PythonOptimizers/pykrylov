__docformat__ = 'restructuredtext'

import numpy as np

from pykrylov.generic import KrylovMethod

class TFQMR( KrylovMethod ):
    """
    A pure Python implementation of the transpose-free quasi-minimal residual
    (TFQMR) algorithm. TFQMR may be used to solve unsymmetric systems of linear
    equations, i.e., systems of the form

        `A x = b`

    where the operator `A` may be unsymmetric.

    TFQMR requires 2 operator-vector products with `A`, 4 dot products and
    10 daxpys per iteration. It does not require products with the adjoint
    of `A`.

    If a preconditioner is supplied, TFQMR needs to solve 2 preconditioning
    systems per iteration. Our implementation is inspired by the original
    description in [Freund]_ and that of [Kelley]_.

    References:

    .. [Freund] R. W. Freund, *A Transpose-Free Quasi-Minimal Residual Method
                for Non-Hermitian Linear Systems*, SIAM Journal on Scientific
                Computing, **14** (2), pp. 470--482, 1993.
    """

    def __init__(self, op, **kwargs):
        KrylovMethod.__init__(self, op, **kwargs)

        self.name = 'Transpose-Free Quasi-Minimum Residual'
        self.acronym = 'TFQMR'
        self.prefix = self.acronym + ': '

    def solve(self, rhs, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the TFQMR method.
        The vector `rhs` should be a Numpy array.

        :keywords:
            :guess:      Initial guess (Numpy array, default: 0)
            :matvec_max: Max. number of matrix-vector produts (2n)
        """
        n = rhs.shape[0]
        nMatvec = 0

        # Initial guess is zero unless one is supplied
        result_type = np.result_type(self.op.dtype, rhs.dtype)
        guess_supplied = 'guess' in kwargs.keys()
        x = kwargs.get('guess', np.zeros(n)).astype(result_type)
        matvec_max = kwargs.get('matvec_max', 2*n)

        r0 = rhs  # Fixed vector throughout
        if guess_supplied:
            r0 = rhs - self.op * x

        rho = np.dot(r0,r0)
        residNorm = np.abs(np.sqrt(rho))
        self.residNorm0 = residNorm
        threshold = max( self.abstol, self.reltol * self.residNorm0 )
        self.logger.info('Initial residual = %8.2e' % self.residNorm0)
        self.logger.info('Threshold = %8.2e' % threshold)

        finished = (residNorm <= threshold or nMatvec >= matvec_max)

        if not finished:
            y = r0.copy()   # Initial residual vector
            w = r0.copy()
            d = np.zeros(n, dtype=result_type)
            theta = 0.0
            eta = 0.0
            k = 0
            if self.precon is not None:
                z = self.precon * y
            else:
                z = y

            u = self.op * z ; nMatvec += 1
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
            c = 1.0 / np.sqrt(1 + theta*theta)
            residNorm *= theta * c
            eta = c * c * alpha
            x += eta * d
            m = 2.0 * k - 1.0
            if residNorm * np.sqrt(m+1) < threshold or nMatvec >= matvec_max:
                finished = True
                continue

            # Second pass
            m += 1
            y -= alpha * v

            if self.precon is not None:
                z = self.precon * y
            else:
                z = y

            u = self.op * z ; nMatvec += 1
            w -= alpha * u
            d *= theta * theta * eta / alpha
            d += z
            theta = np.linalg.norm(w)/residNorm
            c = 1.0 / np.sqrt(1 + theta*theta)
            residNorm *= theta * c
            eta = c * c * alpha
            x += eta * d
            if residNorm * np.sqrt(m+1) < threshold or nMatvec >= matvec_max:
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
                z = self.precon * y
            else:
                z = y

            u = self.op * z ; nMatvec += 1

            # Complete update of v
            v += u

            # Display current info if requested
            self.logger.info('%5d  %8.2e' % (nMatvec, residNorm))


        self.converged = residNorm * np.sqrt(m+1) < threshold
        self.nMatvec = nMatvec
        self.bestSolution = self.x = x
        self.residNorm = residNorm
