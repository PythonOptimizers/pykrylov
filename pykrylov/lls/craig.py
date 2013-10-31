# -*- coding: utf-8 -*-
"""
Solve the least-squares problem

  minimize ||Ax-b||

using CRAIG, or the regularized least-squares problem

  minimize ||Ax-b||^2_D + ||Dx||^2_N

using the generalized CRAIG method, where D and N are symmetric and positive
definite operators.

Dominique Orban, GERAD and Ecole Polytechnique de Montreal
dominique.orban@gerad.ca
"""

from pykrylov.generic import KrylovMethod

from numpy import zeros, dot, inf
from numpy.linalg import norm
from math import sqrt

__docformat__ = 'restructuredtext'

# Simple shortcuts---linalg.norm is too slow for small vectors
def normof2(x,y): return sqrt(x*x + y*y)
def normof4(x1,x2,x3,x4): return sqrt(x1*x1 + x2*x2 + x3*x3 + x4*x4)

class CRAIGFramework(KrylovMethod):
    r"""
    CRAIG solves  `Ax = b`  or  `minimize |b - Ax|` in Euclidian norm  if
    `damp = 0`, or `minimize |b - Ax| + damp * |x|` in Euclidian norm if
    `damp > 0`.

    The generalized CRAIG method solves the regularized linear least-squares
    problem

    minimize |b - Ax|^2_D + |x|^2_N

    for given positive definite matrices D and N of appropriate size.
    Equivalently, solve the symmetric and quasi-definite linear system

    [ M   A ] [ r ]   [ b ]
    [ A' -N ] [ x ] = [ 0 ]

    where M := inv(D).

    `A`  is an (m x n) linear operator defined by  `y = A * x` (or `y = A(x)`),
    where `y` is the result of applying the linear operator to `x`. Application
    of transpose linear operator must be accessible via `u = A.T * x` (or
    `u = A.T(x)`). The shape of the linear operator `A` must be accessible via
    `A.shape`. A convenient way to achieve this is to make sure that `A` is
    a `LinearOperator` instance.

    CRAIG uses an iterative (conjugate-gradient-like) method.

    For further information, see

    1. M. Arioli and D. Orban,
       Iterative Methods for Symmetric Quasi-Definite Linear Systems,
       Cahier du GERAD G-2013-xx, GERAD, Montreal, Canada, 2012.
    2. J. E. Craig, The N-step iteration procedures, Journal of Mathematics and
       Physics, 34(1):64–73, 1955.
    3. J. E. Craig. Iteration Procedures for Simultaneous Equations, Sc.D.
       Thesis, Massachussetts Institute of Technology, 1954.
    """

    def __init__(self, A, **kwargs):

        # Initialize.
        KrylovMethod.__init__(self, A, **kwargs)
        self.name = "CRAIG's Method for Least Squares"
        self.acronym = 'CRAIG'
        self.prefix = self.acronym + ': '

        self.msg=['The exact solution is  x = 0                              ',
                  'Ax - b is small enough, given atol, btol                  ',
                  'The least-squares solution is good enough, given atol     ',
                  'The estimate of cond(Abar) has exceeded conlim            ',
                  'Ax - b is small enough for this machine                   ',
                  'The least-squares solution is good enough for this machine',
                  'Cond(Abar) seems to be too large for this machine         ',
                  'The iteration limit has been reached                      ',
                  'The truncated direct error is small enough, given etol    ']

        self.A = A
        self.x = None ; self.var = None

        self.itn = 0; self.istop = 0
        self.Anorm = 0.; self.Acond = 0. ; self.Arnorm = 0.
        self.xnorm = 0.;
        self.r1norm = 0.; self.r2norm = 0.
        self.optimal = False
        self.norms  = []    # Squared energy norm of iterates.
        self.resids = []    # Squared least-squares objective function values.
        self.normal_eqns_resids = [] # Resids of normal equations (not squared).
        self.dir_errors_p_window = []
        self.dir_errors_d_window = []
        self.iterates_p = []  # Primal iterates.
        self.iterates_d = []  # Dual   iterates.
        return

    def solve(self, rhs, itnlim=0, damp=0.0, M=None, N=None, atol=1.0e-9,
              btol=1.0e-9, conlim=1.0e+8, show=False, wantvar=False, **kwargs):
        """
        Solve the linear system, linear least-squares problem or regularized
        linear least-squares problem with specified parameters. All return
        values below are stored in members of the same name.

        :parameters:

           :rhs:    right-hand side vector.
           :itnlim: is an explicit limit on iterations (for safety).
           :damp:   damping/regularization parameter.
           :M:      inv(D) as a linear operator if not the identity operator.
           :N:      inv(C) as a linear operator if nonzero. If specified,
                    `damp` is automatically reset to 1.

        :keywords:

           :atol:
           :btol:  are stopping tolerances.  If both are 1.0e-9 (say),
                   the final residual norm should be accurate to about 9 digits.
                   (The final x will usually have fewer correct digits,
                   depending on `cond(A)` and the size of `damp`.)
           :etol:  stopping tolerance based on direct error (default 1.0e-6).
           :conlim: is also a stopping tolerance.  lsqr terminates if an
                    estimate of `cond(A)` exceeds `conlim`.  For compatible
                    systems `Ax = b`, `conlim` could be as large as 1.0e+12
                    (say).  For least-squares problems, `conlim` should be less
                    than 1.0e+8. Maximum precision can be obtained by setting
                    `atol` = `btol` = `conlim` = zero, but the number of
                    iterations may then be excessive.
           :show:   if set to `True`, gives an iteration log.
                    If set to `False`, suppresses output.
           :store_resids: Store full residual norm history (default: False)
           :window: Number of consecutive iterations over which the director error
                    should be measured (default: 5).

        :return:

           :x:     is the final solution.
           :istop: gives the reason for termination.
           :istop: = 1 means x is an approximate solution to Ax = b.
                   = 2 means x approximately solves the least-squares problem.
           :r1norm: = norm(r), where r = b - Ax.
           :r2norm: = sqrt(norm(r)^2  +  damp^2 * norm(x)^2)
                    = r1norm if damp = 0.
           :Anorm: = estimate of Frobenius norm of (regularized) A.
           :Acond: = estimate of cond(Abar).
           :Arnorm: = estimate of norm(A'r - damp^2 x).
           :xnorm: = norm(x).
           :var:   (if present) estimates all diagonals of (A'A)^{-1}
                   (if damp=0) or more generally (A'A + damp^2*I)^{-1}.
                   This is well defined if A has full column rank or damp > 0.
                   (Not sure what var means if rank(A) < n and damp = 0.)
        """

        etol = kwargs.get('etol', 1.0e-6)
        store_resids = kwargs.get('store_resids', False)
        store_iterates = kwargs.get('store_iterates', False)
        window = kwargs.get('window', 5)

        self.norms  = []    # Squared energy norm of iterates.
        self.resids = []    # Squared least-squares objective function values.
        self.normal_eqns_resids = [] # Resids of normal equations (not squared).
        self.dir_errors_p_window = []
        self.dir_errors_d_window = []
        self.iterates_p = []  # Primal iterates.
        self.iterates_d = []  # Dual   iterates.

        A = self.A
        m, n = A.shape

        if itnlim == 0: itnlim = 3*n

        if wantvar:
            var = zeros(n,1)
        else:
            var = None

        if N is not None:
            damp = dampsq = 1.0
        else:
            dampsq = damp*damp;

        itn = istop = 0
        ctol = 0.0
        if conlim > 0.0: self.ctol = 1.0/conlim
        #Anorm = Acond = 0.
        #z = xnorm = xxnorm = ddnorm = res2 = 0.
        #cs2 = -1. ; sn2 = 0.
        #xnorm = 0.0

        if show:
            print ' '
            print 'CRAIG           Least-squares solution of  Ax = b'
            str1='The matrix A has %8d rows and %8d cols' % (m, n)
            str2='damp = %20.14e     wantvar = %-5s' % (damp, repr(wantvar))
            str3='atol = %8.2e                 conlim = %8.2e' % (atol,conlim)
            str4='btol = %8.2e                 itnlim = %8g' % (btol, itnlim)
            print str1; print str2; print str3; print str4;

        # Set up the first vectors u and v for the bidiagonalization.
        # These satisfy  beta*M*u = b,  alpha*N*v = A'u.

        r = zeros(m)
        x = zeros(n)
        rNrgNorm2 = 0.0
        xNrgNorm2 = 0.0          # Squared energy norm of final solution.
        dErr = zeros(window)     # Truncated direct error terms.
        trncDirErr = 0           # Truncated direct error.

        Mu = rhs[:m].copy()
        if M is not None:
            u = M(Mu)
        else:
            u = Mu

        alpha = 0.
        beta = sqrt(dot(u,Mu))       # norm(u)
        if beta > 0:
            u /= beta
            if M is not None: Mu /= beta

            Nv = A.T * u
            if N is not None:
                v = N(Nv)
            else:
                v = Nv
            alpha = sqrt(dot(v,Nv))   # norm(v)

        if alpha > 0:
            v /= alpha
            if N is not None: Nv /= alpha

        x_is_zero = False   # Is x=0 the solution to the least-squares prob?
        #Arnorm = alpha * beta
        #if Arnorm == 0.0:
        if beta == 0.0:
            if show: print self.msg[0]
            x_is_zero = True
            istop = 0

        bnorm = beta
        delta = 1.
        rho   = normof2(alpha, 1)

        # Dual variable initialization.
        d     = u / rho
        tau   = beta / rho
        r     = tau * d
        rnorm = tau * tau

        # Primal variable initialization.
        c     = alpha / rho
        s     = 1 / rho
        zeta  = s * beta
        eta   = c * zeta    # Most recently computed component of \bar{y}.
        xi    = s * zeta    # Most recently computed component of \bar{x}.
        w     = c * v
        wbar  = s * v
        x     = zeta * w
        xnorm = eta * eta

        #rnorm  = zeta * zeta    # = ||b - Ax||^2_M + \|x\|^2_N.
        r1norm = xi * xi        # = ||b - Ax||^2_M.
        r2norm = rnorm

        head1  = '   Itn      x(1)       r1norm     r2norm '
        head2  = ' Compatible   LS      Norm A   Cond A'

        if store_iterates:
            self.iterates_p.append(x.copy())
            self.iterates_d.append(r.copy())

        #if show:
        #    print ' '
        #    print head1+head2
        #    test1  = 1.0
        #    test2  = alpha / beta if not x_is_zero else 1.0
        #    str1   = '%6g %12.5e'     % (itn,    x[0])
        #    str2   = ' %10.3e %10.3e' % (r1norm, r2norm)
        #    str3   = '  %8.1e %8.1e'  % (test1,  test2)
        #    print str1+str2+str3

        if store_resids:
            self.norms.append(xNrgNorm2)
            self.resids.append(r2norm)
        #    self.normal_eqns_resids.append(Arnorm)

        # ------------------------------------------------------------------
        #     Main iteration loop.
        # ------------------------------------------------------------------
        while itn < itnlim and not x_is_zero:

            itn = itn + 1

            #   Perform the next step of the bidiagonalization to obtain the
            #   next  beta, u, alpha, v.  These satisfy the relations
            #               beta*M*u  =  A*v   -  alpha*M*u,
            #              alpha*N*v  =  A'*u  -   beta*N*v.

            Mu = A*v - alpha*Mu
            if M is not None:
                u = M(Mu)
            else:
                u = Mu
            beta = sqrt(dot(u,Mu))   # norm(u)

            # Update residual of CRAIG's "other" normal equations.
            Arnorm = abs(alpha * beta * s * zeta)

            # Update residual estimate of SQD system.
            sqd_resid = beta * c * zeta

            if beta > 0:
                u /= beta
                if M is not None: Mu /= beta

                #Anorm = normof4(Anorm, alpha, beta, damp)

                Nv = A.T*u - beta*Nv
                if N is not None:
                    v = N(Nv)
                else:
                    v = Nv
                alpha = sqrt(dot(v,Nv))  # norm(v)
                if alpha > 0:
                    v /= alpha
                    if N is not None: Nv /= alpha

            # Perform rotation of type I and compute rotation of type II.

            beta_hat = c * beta
            gamma    = s * beta
            delta    = normof2(gamma, 1)

            c2       = -1 / delta
            s2       = gamma / delta

            # Perform rotation of type II and compute next rotation of type I.

            alpha_hat = normof2(alpha, delta)
            c         = alpha / alpha_hat
            s         = delta / alpha_hat

            # Update dual variables.
            d = (u - beta_hat * d) / alpha_hat
            tau = -beta_hat * tau / alpha_hat
            r += tau * d

            # Update x, w and wbar.

            zeta = - beta_hat * zeta / alpha_hat
            eta  = c * zeta
            xi   = s * zeta

            wbar *= s2
            w     = c * v + s * wbar
            wbar *= -c
            wbar += s * v
            x    += zeta * w

            if store_iterates:
                self.iterates_p.append(x.copy())
                self.iterates_d.append(r.copy())

            #ddnorm  = ddnorm + norm(dk)**2
            #if wantvar: var += dk*dk

            # Update energy norm of x.
            rNrgNorm2 += tau * tau
            xNrgNorm2 += zeta * zeta             # (A + B*inv(C)*B')-norm of x.
            dErr[itn % window] = tau  #zeta
            if itn > window:
                trncDirErr = norm(dErr)
                rNrgNorm = sqrt(rNrgNorm2)
                self.dir_errors_d_window.append(trncDirErr / rNrgNorm)
                if trncDirErr < etol * rNrgNorm:
                    istop = 8

            # Use a plane rotation on the right to eliminate the
            # super-diagonal element (theta) of the upper-bidiagonal matrix.
            # Then use the result to estimate norm(x).

            #delta   =   sn2 * rho
            #gambar  = - cs2 * rho
            #rhs     =   phi  -  delta * z
            #zbar    =   rhs / gambar
            #xnorm   =   sqrt(xxnorm + zbar**2)
            #rnorm   += zeta * zeta               # inv(A)-norm of ||b-Ax||.
            rnorm   += tau * tau
            xnorm   += eta * eta                 # C-norm of x.
            #gamma   =   normof2(gambar, theta)
            #cs2     =   gambar / gamma
            #sn2     =   theta  / gamma
            #z       =   rhs    / gamma
            #xxnorm +=   z*z

            # Test for convergence.
            # First, estimate the condition of the matrix  Abar,
            # and the norms of  rbar  and  Abar'rbar.

            #Acond   =   Anorm * sqrt(ddnorm)
            #res1    =   phibar**2
            #res2    =   res2  +  psi**2
            #Arnorm  =   alpha * abs(tau)

            # 07 Aug 2002:
            # Distinguish between
            #    r1norm = ||b - Ax|| and
            #    r2norm = rnorm in current code
            #           = sqrt(r1norm^2 + damp^2*||x||^2).
            #    Estimate r1norm from
            #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
            # Although there is cancellation, it might be accurate enough.

            #r1sq    =   rnorm**2  -  dampsq * xxnorm
            r1norm += xi * xi
            #if r1sq < 0: r1norm = - r1norm
            r2norm = rnorm

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.

            test1 = sqrt(rnorm) / bnorm
            #if Anorm == 0. or rnorm == 0.:
            #    test2 = inf
            #else:
            #    test2 = Arnorm/(Anorm * rnorm)
            #if Acond == 0.0:
            #    test3 = inf
            #else:
            #    test3 = 1.0 / Acond
            t1    = test1 #/ (1    +  Anorm * xnorm / bnorm)
            rtol  = btol  #+  atol *  Anorm * xnorm / bnorm

            if store_resids:
                self.norms.append(xNrgNorm2)
                self.resids.append(r2norm)
                self.normal_eqns_resids.append(Arnorm)

            # The following tests guard against extremely small values of
            # atol, btol  or  ctol.  (The user may have set any or all of
            # the parameters  atol, btol, conlim  to 0.)
            # The effect is equivalent to the normal tests using
            # atol = eps,  btol = eps,  conlim = 1/eps.

            if itn >= itnlim:  istop = 7
            #if 1 + test3 <= 1: istop = 6
            #if 1 + test2 <= 1: istop = 5
            if 1 + t1    <= 1: istop = 4

            # Allow for tolerances set by the user.

            #if test3 <= ctol: istop = 3
            #if test2 <= atol: istop = 2
            if test1 <= rtol: istop = 1

            # See if it is time to print something.

            prnt = False;
            if n     <= 40       : prnt = True
            if itn   <= 10       : prnt = True
            if itn   >= itnlim-10: prnt = True
            if itn % 10 == 0     : prnt = True
            #if test3 <=  2*ctol  : prnt = True
            #if test2 <= 10*atol  : prnt = True
            #if test1 <= 10*rtol  : prnt = True
            if istop !=  0       : prnt = True

            if prnt and show:
                str1 = '%6g %12.5e'     % (  itn,   x[0])
                #str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
                #str3 = '  %8.1e %8.1e'  % (test1,  test2)
                #str4 = ' %8.1e %8.1e'   % (Anorm,  Acond)
                #print str1+str2+str3+str4

            if istop > 0: break

            # End of iteration loop.
            # Print the stopping condition.

        if show:
            print ' '
            print 'CRAIG finished'
            print self.msg[istop]
            print ' '
            str1 = 'istop =%8g   r1norm =%8.1e'   % (istop, sqrt(r1norm))
            #str2 = 'Anorm =%8.1e   Arnorm =%8.1e' % (Anorm, Arnorm)
            str3 = 'itn   =%8g   r2norm =%8.1e'   % ( itn, sqrt(r2norm))
            #str4 = 'Acond =%8.1e   xnorm  =%8.1e' % (Acond, xnorm )
            str5 = '                  bnorm  =%8.1e'    % bnorm
            str6 = 'xNrgNorm2 = %7.1e   trnDirErr = %7.1e' % \
                    (xNrgNorm2, trncDirErr)
            print str1 #+ '   ' + str2
            print str3 #+ '   ' + str4
            print str5
            print str6
            print ' '

        if istop == 0: self.status = 'solution is zero'
        if istop in [1,2,4,5]: self.status = 'residual small'
        if istop in [3,6]: self.status = 'ill-conditioned operator'
        if istop == 7: self.status = 'max iterations'
        if istop == 8: self.status = 'direct error small'
        self.optimal = istop in [1,2,4,5,8]
        self.x = self.bestSolution = x
        self.r = r
        self.istop = istop
        self.itn = itn
        self.nMatvec = 2*itn
        self.r1norm = sqrt(r1norm)
        self.r2norm = sqrt(r2norm)
        #self.residNorm = r2norm
        #self.Anorm = Anorm
        #self.Acond = Acond
        self.Arnorm = Arnorm
        self.xnorm = xnorm
        #self.var = var
        return


if __name__ == '__main__':

    # Solve the SQD system
    #  [ 2  1 ] [x] = [2]
    #  [ 1 -3 ] [y]   [0]
    # Solution: x = 6/7 = 0.857142857143
    #           y = 2/7 = 0.285714285714

    from pykrylov.linop import LinearOperator
    import numpy as np

    A = LinearOperator(1, 1, matvec=lambda u: u/2, symmetric=True)
    C = LinearOperator(1, 1, matvec=lambda v: v/3, symmetric=True)
    B = LinearOperator(1, 1, matvec=lambda x: x.copy(), symmetric=True)
    rhs = np.array([2.0])
    craig = CRAIGFramework(B)
    craig.solve(rhs, M=A, N=C, show=True)
    print 'Solution: ', craig.x
