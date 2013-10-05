
__docformat__ = 'restructuredtext'

import numpy as np


from pykrylov.generic import KrylovMethod

class BiCGSTAB( KrylovMethod ):
    """
    A pure Python implementation of the bi-conjugate gradient stabilized
    (Bi-CGSTAB) algorithm. Bi-CGSTAB may be used to solve unsymmetric systems
    of linear equations, i.e., systems of the form

        A x = b

    where the operator A is unsymmetric and nonsingular.

    Bi-CGSTAB requires 2 operator-vector products, 6 dot products and 6 daxpys
    per iteration.

    In addition, if a preconditioner is supplied, it needs to solve 2
    preconditioning systems per iteration.

    The original description appears in [VdVorst92]_. Our implementation is a
    preconditioned version of that given in [Kelley]_.

    Reference:

    .. [VdVorst92] H. Van der Vorst, *Bi-CGSTAB: A Fast and Smoothly Convergent
                   Variant of Bi-CG for the Solution of Nonsymmetric Linear
                   Systems*, SIAM Journal on Scientific and Statistical
                   Computing **13** (2), pp. 631--644, 1992.
    """

    def __init__(self, op, **kwargs):
        KrylovMethod.__init__(self, op, **kwargs)

        self.name = 'Bi-Conjugate Gradient Stabilized'
        self.acronym = 'Bi-CGSTAB'
        self.prefix = self.acronym + ': '

    def solve(self, rhs, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the Bi-CGSTAB
        method. The vector `rhs` should be a Numpy array.

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

        # Initial residual is the fixed vector
        r0 = rhs
        if guess_supplied:
            r0 = rhs - self.op * x
            nMatvec += 1

        rho = alpha = omega = 1.0
        rho_next = np.dot(r0,r0)
        residNorm = self.residNorm0 = np.abs(np.sqrt(rho_next))
        threshold = max( self.abstol, self.reltol * self.residNorm0 )

        finished = (residNorm <= threshold or nMatvec >= matvec_max)

        self.logger.info('Initial residual = %8.2e' % self.residNorm0)
        self.logger.info('Threshold = %8.2e' % threshold)
        hdr = '%6s  %8s' % ('Matvec', 'Residual')
        self.logger.info(hdr)
        self.logger.info('-' * len(hdr))

        if not finished:
            r = r0.copy()
            p = np.zeros(n, dtype=result_type)
            v = np.zeros(n, dtype=result_type)

        while not finished:

            beta = rho_next/rho * alpha/omega
            rho = rho_next

            # Update p in-place
            p *= beta
            p -= beta * omega * v
            p += r

            # Compute preconditioned search direction
            if self.precon is not None:
                q = self.precon * p
            else:
                q = p

            v = self.op * q ; nMatvec += 1

            alpha = rho/np.dot(r0, v)
            s = r - alpha * v

            # Check for CGS termination
            residNorm = np.linalg.norm(s)

            self.logger.info('%6d  %8.2e' % (nMatvec, residNorm))

            if residNorm <= threshold:
                x += alpha * q
                finished = True
                continue

            if nMatvec >= matvec_max:
                finished = True
                continue

            if self.precon is not None:
                z = self.precon * s
            else:
                z = s

            t = self.op * z ; nMatvec += 1
            omega = np.dot(t,s)/np.dot(t,t)
            rho_next = -omega * np.dot(r0,t)

            # Update residual
            r = s - omega * t

            # Update solution in-place-ish. Note that 'z *= omega' alters s if
            # precon = None. That's ok since s is no longer needed in this iter.
            # 'q *= alpha' would alter p.
            z *= omega
            x += z
            x += alpha * q

            residNorm = np.linalg.norm(r)

            self.logger.info('%6d  %8.2e' % (nMatvec, residNorm))

            if residNorm <= threshold or nMatvec >= matvec_max:
                finished = True
                continue


        self.converged = residNorm <= threshold
        self.nMatvec = nMatvec
        self.bestSolution = self.x = x
        self.residNorm = residNorm
