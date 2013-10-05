"""
A Python implementation of SYMMLQ.

This is a line-by-line translation from Matlab code
available at http://www.stanford.edu/group/SOL/software/symmlq.htm.

.. moduleauthor: D. Orban <dominique.orban@gerad.ca>
"""

__docformat__ = 'restructuredtext'

import numpy as np

from pykrylov.generic import KrylovMethod
from pykrylov.tools   import machine_epsilon

class Symmlq(KrylovMethod) :
    """
    SYMMLQ is designed to solve the system of linear equations A x = b
    where A is an n by n symmetric operator and b is a given vector.

    If shift is nonzero, SYMMLQ solves (A - shift I) x = b.

    SYMMLQ requires one operator-vector products with `A`, 2 dot products and
    4 daxpys per iteration.

    If a preconditioner is supplied, SYMMLQ needs to solve one preconditioning
    system per iteration. This is a Pythonized line-by-line translation of
    Michael Saunders' `original SYMMLQ implementation in Matlab
    <http://www.stanford.edu/group/SOL/software/symmlq.htm>`_


    :parameters:

            :op:  an operator describing the coefficient operator `A`.
                  `y = op * x` must return the operator-vector product
                  `y = Ax` for any given vector `x`.

    :keywords:

        :precon:  optional preconditioner. If not `None`, `y = precon * x`
                  returns the vector `y` solution of the linear system
                  `M y = x`. The preconditioner must be symmetric and
                  positive definite.


    Note: `atol` has no effect in this method.

    References:

    .. [PaiSau75] C. C. Paige and M. A. Saunders, *Solution of Sparse Indefinite
                  Systems of Linear Equations*, SIAM Journal on Numerical
                  Analysis, **12** (4), pp. 617--629, 1975.
    """

    def __init__(self, op, **kwargs):
        KrylovMethod.__init__(self, op, **kwargs)

        self.name = 'Symmetric Indefinite Lanczos with Orthogonal Factorization'
        self.acronym = 'SYMMLQ'
        self.prefix = self.acronym + ': '
        self.iterates = []


    def solve(self, rhs, **kwargs):
        """
        Solve a linear system with `rhs` as right-hand side by the SYMMLQ
        method.

        :parameters:

            :rhs:     right-hand side vector b. Should be a Numpy array.

        :keywords:

            :matvec_max: Max. number of matrix-vector produts. Default: 2n+2.
            :rtol:    relative stopping tolerance. Default: 1.0e-9.
            :shift:   optional shift value. Default: 0.
            :check:   specify whether or not to check that `matvec` indeed
                      describes a symmetric matrix and that `precon` indeed
                      describes a symmetric positive-definite preconditioner.
        """

        # Set parameters
        n = rhs.shape[0]
        nMatvec = 0

        matvec_max = kwargs.get('matvec_max', 2*n+2)
        rtol = kwargs.get('rtol', 1.0e-9)
        check = kwargs.get('check', False)
        shift = kwargs.get('shift', None)
        if shift == 0.0: shift = None
        eps = machine_epsilon()
        store_iterates = kwargs.get('store_iterates', False)

        first = 'Enter SYMMLQ.   '
        last  = 'Exit  SYMMLQ.   '
        space = ' '
        msg={
            -1:' beta2 = 0.  If M = I, b and x are eigenvectors',
             0:' beta1 = 0.  The exact solution is  x = 0',
             1:' Requested accuracy achieved, as determined by rtol',
             2:' Reasonable accuracy achieved, given eps',
             3:' x has converged to an eigenvector',
             4:' acond has exceeded 0.1/eps',
             5:' The iteration limit was reached',
             6:' aprod  does not define a symmetric matrix',
             7:' msolve does not define a symmetric matrix',
             8:' msolve does not define a pos-def preconditioner'}

        self.logger.info(first + 'Solution of symmetric Ax = b')
        fmt = 'n     =  %3g    precon =  %5s           '
        self.logger.info(fmt % (n, repr(self.precon is None)))
        if shift is not None: self.logger.info('shift  =  %23.14e' % shift)
        fmt = 'maxit =  %3g     eps    =  %11.2e    rtol   =  %11.2e'
        self.logger.info(fmt % (int((matvec_max-2.0)/2),eps,rtol))

        istop  = 0 ; ynorm  = 0 ; w = np.zeros(n) ; acond = 0
        itn    = 0 ; xnorm  = 0 ; x = np.zeros(n) ; done=False
        anorm  = 0 ; rnorm  = 0 ; v = np.zeros(n)

        if store_iterates:
            self.iterates.append(x.copy())

        # Set up y for the first Lanczos vector v1.
        # y is really beta1 * P * v1  where  P = C^(-1).
        # y and beta1 will be zero if b = 0.

        r1 = rhs.copy()
        if self.precon is not None:
            y = self.precon * r1
        else:
            y = rhs.copy()
        b1 = y[0] ; beta1 = np.dot(r1, y)

        # Ensure preconditioner is symmetric.

        if check and self.precon is not None:
            r2 = self.precon * y
            s = np.dot(y,y)
            t = np.dot(r1,r2)
            z = np.abs(s-t)
            epsa = (s+eps) * eps**(1.0/3)
            if z > epsa:
                istop = 7
                done = True

        # Test for an indefinite preconditioner.
        # If rhs = 0 exactly, stop with x = 0.

        if beta1 <  0:
            istop = 8
            done = True
        if beta1 == 0:
            done = True

        if beta1 > 0:
            beta1 = np.sqrt(beta1)
            s     = 1.0 / beta1
            v     = s * y

            y = self.matvec(v) ; nMatvec += 1
            if check:
                r2 = self.op * y  # Do not count this matrix-vector product
                s = np.dot(y,y)
                t = np.dot(v,r2)
                z = abs(s-t)
                epsa = (s+eps) * eps**(1.0/3)
                if z > epsa:
                    istop = 6
                    done = True

            # Set up y for the second Lanczos vector.
            # Again, y is beta * P * v2  where  P = C^(-1).
            # y and beta will be zero or very small if Abar = I or constant * I.

            if shift is not None: y -= shift * v
            alfa = np.dot(v, y)
            y -= (alfa / beta1) * r1

            # Make sure  r2  will be orthogonal to the first  v.

            z  = np.dot(v, y)
            s  = np.dot(v, v)
            y -= (z / s) * v
            r2 = y.copy()

            if self.precon is not None: y = self.precon * r2
            oldb   = beta1
            beta   = np.dot(r2, y)
            if beta < 0:
                istop = 8
                done = True

            #  Cause termination (later) if beta is essentially zero.

            beta = np.sqrt(beta)
            if beta <= eps:
                istop = -1

            #  See if the local reorthogonalization achieved anything.

            denom = np.sqrt(s) * np.linalg.norm(r2) + eps
            s = z / denom
            t = np.dot(v, r2)
            t = t / denom

            self.logger.info('beta1 =  %10.2e   alpha1 =  %9.2e'% (beta1,alfa))
            self.logger.info('(v1, v2) before and after  %14.2e' % s)
            self.logger.info('local reorthogonalization  %14.2e' % t)

            #  Initialize other quantities.
            cgnorm = beta1 ; rhs2   = 0 ; tnorm  = alfa**2 + beta**2
            gbar   = alfa  ; bstep  = 0 ; ynorm2 = 0
            dbar   = beta  ; snprod = 1 ; gmax   = np.abs(alfa) + eps
            rhs1   = beta1 ; x1cg   = 0 ; gmin   = gmax
            qrnorm = beta1

        # end  if beta1 > 0

        head1 = '   Itn     x(1)(cg)  normr(cg)  r(minres)'
        head2 = '    bstep    anorm    acond'
        self.logger.info(head1 + head2)

        str1 = '%6g %12.5e %10.3e' % (itn, x1cg, cgnorm)
        str2 = ' %10.3e  %8.1e' %    (qrnorm, bstep/beta1)
        self.logger.info(str1 + str2)

        # ------------------------------------------------------------------
        # Main iteration loop.
        # ------------------------------------------------------------------
        # Estimate various norms and test for convergence.

        if not done:
            while nMatvec < matvec_max:
                itn    = itn  +  1
                anorm  = np.sqrt(tnorm)
                ynorm  = np.sqrt(ynorm2)
                epsa   = anorm * eps
                epsx   = anorm * ynorm * eps
                epsr   = anorm * ynorm * rtol
                diag   = gbar

                if diag == 0: diag = epsa

                lqnorm = np.sqrt(rhs1**2 + rhs2**2)
                qrnorm = snprod * beta1
                cgnorm = qrnorm * beta / np.abs(diag)

                # Estimate  Cond(A).
                # In this version we look at the diagonals of  L  in the
                # factorization of the tridiagonal matrix,  T = L*Q.
                # Sometimes, T(k) can be misleadingly ill-conditioned when
                # T(k+1) is not, so we must be careful not to overestimate acond

                if lqnorm < cgnorm:
                    acond  = gmax / gmin
                else:
                    denom  = min(gmin, np.abs(diag))
                    acond  = gmax / denom

                zbar = rhs1 / diag
                z    = (snprod * zbar + bstep) / beta1
                x1lq = x[0] + b1 * bstep / beta1
                x1cg = x[0] + w[0] * zbar  +  b1 * z

                # See if any of the stopping criteria are satisfied.
                # In rare cases, istop is already -1 from above
                # (Abar = const * I).

                if istop == 0:
                    if nMatvec >= matvec_max : istop = 5
                    if acond   >= 0.1/eps    : istop = 4
                    if epsx    >= beta1      : istop = 3
                    if cgnorm  <= epsx       : istop = 2
                    if cgnorm  <= epsr       : istop = 1

                prnt = False
                if n       <= 40              :   prnt = True
                if nMatvec <= 20              :   prnt = True
                if nMatvec >= matvec_max - 10 :   prnt = True
                if itn%10 == 0                :   prnt = True
                if cgnorm <= 10.0*epsx        :   prnt = True
                if cgnorm <= 10.0*epsr        :   prnt = True
                if acond  >= 0.01/eps         :   prnt = True
                if istop  != 0                :   prnt = True

                if prnt:
                    str1 =  '%6g %12.5e %10.3e' % (itn, x1cg, cgnorm)
                    str2 =  ' %10.3e  %8.1e' %    (qrnorm, bstep/beta1)
                    str3 =  ' %8.1e %8.1e' %      (anorm, acond)
                    self.logger.info(str1 + str2 + str3)

                if istop !=0:
                    break

                # Obtain the current Lanczos vector  v = (1 / beta)*y
                # and set up  y  for the next iteration.

                s = 1/beta
                v = s * y
                y = self.op * v ; nMatvec += 1
                if shift is not None: y -= shift * v
                y -= (beta / oldb) * r1
                alfa = np.dot(v, y)
                y -= (alfa / beta) * r2
                r1 = r2.copy()
                r2 = y.copy()
                if self.precon is not None: y = self.precon * r2
                oldb = beta
                beta = np.dot(r2, y)

                if beta < 0:
                    istop = 6
                    break

                beta  = np.sqrt(beta)
                tnorm = tnorm  +  alfa**2  +  oldb**2  +  beta**2

                # Compute the next plane rotation for Q.

                gamma  = np.sqrt(gbar**2 + oldb**2)
                cs     = gbar / gamma
                sn     = oldb / gamma
                delta  = cs * dbar  +  sn * alfa
                gbar   = sn * dbar  -  cs * alfa
                epsln  = sn * beta
                dbar   =            -  cs * beta

                # Update  X.

                z = rhs1 / gamma
                s = z*cs
                t = z*sn
                x += s*w + t*v
                w *= sn ; w -= cs*v

                if store_iterates:
                    self.iterates.append(x.copy())

                # Accumulate the step along the direction b, and go round again.

                bstep  = snprod * cs * z  +  bstep
                snprod = snprod * sn
                gmax   = max(gmax, gamma)
                gmin   = min(gmin, gamma)
                ynorm2 = z**2  +  ynorm2
                rhs1   = rhs2  -  delta * z
                rhs2   =       -  epsln * z
            # end while

        # ------------------------------------------------------------------
        # End of main iteration loop.
        # ------------------------------------------------------------------

        # Move to the CG point if it seems better.
        # In this version of SYMMLQ, the convergence tests involve
        # only cgnorm, so we're unlikely to stop at an LQ point,
        # EXCEPT if the iteration limit interferes.

        if cgnorm < lqnorm:
            zbar   = rhs1 / diag
            bstep  = snprod * zbar + bstep
            ynorm  = np.sqrt(ynorm2 + zbar**2)
            x     += zbar * w

        # Add the step along b.

        bstep  = bstep / beta1
        if self.precon is not None:
            y = self.precon * rhs
        else:
            y = rhs.copy()
        x += bstep * y

        # Compute the final residual,  r1 = b - (A - shift*I)*x.

        y = self.op * x ; nMatvec += 1
        if shift is not None: y -= shift * x
        r1 = rhs - y
        rnorm = np.linalg.norm(r1)
        xnorm = np.linalg.norm(x)

        # ==================================================================
        # Display final status.
        # ==================================================================

        fmt = ' istop   =  %3g               itn   =   %5g'
        self.logger.info(last + fmt % (istop, itn))
        fmt = ' anorm   =  %12.4e      acond =  %12.4e'
        self.logger.info(last + fmt % (anorm, acond))
        fmt = ' rnorm   =  %12.4e      xnorm =  %12.4e'
        self.logger.info(last + fmt % (rnorm, xnorm))
        self.logger.info(last + msg[istop])

        self.nMatvec = nMatvec
        self.bestSolution = x ; self.solutionNorm = xnorm
        self.x = self.bestSolution ; self.xNorm = xnorm
        self.residNorm = rnorm
        self.acond = acond ; self.anorm = anorm
