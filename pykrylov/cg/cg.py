
__docformat__ = 'restructuredtext'

import numpy as np

from pykrylov.tools.utils import check_symmetric
from pykrylov.generic import KrylovMethod

class CG( KrylovMethod ):
    """
    A pure Python implementation of the conjugate gradient (CG) algorithm. The
    conjugate gradient algorithm may be used to solve symmetric positive
    definite systems of linear equations, i.e., systems of the form

        A x = b

    where the operator A is square, symmetric and positive definite. This is
    equivalent to solving the unconstrained convex quadratic optimization
    problem

        minimize    -<b,x> + 1/2 <x, Ax>

    in the variable x.

    CG performs 1 operator-vector product, 2 dot products and 3 daxpys per
    iteration.

    If a preconditioner is supplied, it needs to solve one preconditioning
    system per iteration. Our implementation is standard and follows [Kelley]_
    and [Templates]_.
    """

    def __init__(self, op, **kwargs):
        KrylovMethod.__init__(self, op, **kwargs)

        self.name = 'Conjugate Gradient'
        self.acronym = 'CG'
        self.prefix = self.acronym + ': '
        self.resids = []
        self.iterates = []

        # Direction of nonconvexity if A is not positive definite
        self.infiniteDescent = None


    def solve(self, rhs, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the CG method.
        The vector `rhs` should be a Numpy array.

        :Keywords:

           :guess:           Initial guess (Numpy array). Default: 0.
           :matvec_max:      Max. number of operator-vector produts. Default: 2n.
           :check_symmetric: Ensure operator is symmetric. Default: False.
           :check_curvature: Ensure operator is positive definite. Default: True.
           :store_resids:    Store full residual vector history. Default: False.
           :store_iterates:  Store full iterate history. Default: False.

        """
        n = rhs.shape[0]
        nMatvec = 0
        definite = True
        check_sym = kwargs.get('check_symmetric', False)
        check_curvature = kwargs.get('check_curvature', True)
        store_resids = kwargs.get('store_resids', False)
        store_iterates = kwargs.get('store_iterates', False)

        if check_sym:
            if not check_symmetric(self.op):
                self.logger.error('Coefficient operator is not symmetric')
                return

        # Initial guess
        result_type = np.result_type(self.op.dtype, rhs.dtype)
        guess_supplied = 'guess' in kwargs.keys()
        x = kwargs.get('guess', np.zeros(n)).astype(result_type)

        if store_iterates:
            self.iterates.append(x.copy())

        matvec_max = kwargs.get('matvec_max', 2*n)

        # Initial residual vector
        r = -rhs
        if guess_supplied:
            r += self.op * x
            nMatvec += 1

        # Initial preconditioned residual vector
        if self.precon is not None:
            y = self.precon * r
        else:
            y = r

        if store_resids:
            self.resids.append(y.copy())

        ry = np.dot(r,y)
        self.residNorm0 = residNorm = np.abs(np.sqrt(ry))
        self.residHistory.append(self.residNorm0)
        threshold = max(self.abstol, self.reltol * self.residNorm0)

        p = -r   # Initial search direction (copy not to overwrite rhs if x=0)

        hdr_fmt = '%6s  %7s  %8s'
        hdr = hdr_fmt % ('Matvec', 'Resid', 'Curv')
        self.logger.info(hdr)
        self.logger.info('-' * len(hdr))
        info = '%6d  %7.1e' % (nMatvec, residNorm)
        self.logger.info(info)

        while residNorm > threshold and nMatvec < matvec_max and definite:

            Ap  = self.op * p
            nMatvec += 1
            pAp = np.dot(p, Ap)

            if check_curvature:
                if np.imag(pAp) > 1.0e-8 * np.abs(pAp) or np.real(pAp) <= 0:
                    self.logger.error('Coefficient operator is not positive definite')
                    self.infiniteDescent = p
                    definite = False
                    continue

            # Compute step length
            alpha = ry/pAp

            # Update estimate and residual
            x += alpha * p
            r += alpha * Ap

            if store_iterates:
                self.iterates.append(x.copy())

            # Compute preconditioned residual
            if self.precon is not None:
                y = self.precon * r
            else:
                y = r

            if store_resids:
                self.resids.append(y.copy())

            # Update preconditioned residual norm
            ry_next = np.dot(r,y)

            # Update search direction
            beta = ry_next/ry
            p *= beta
            p -= r

            ry = ry_next
            residNorm = np.abs(np.sqrt(ry))
            self.residHistory.append(residNorm)

            info = '%6d  %7.1e  %8.1e' % (nMatvec, residNorm, np.real(pAp))
            self.logger.info(info)


        self.converged = residNorm <= threshold
        self.definite = definite
        self.nMatvec = nMatvec
        self.bestSolution = self.x = x
        self.residNorm = residNorm
