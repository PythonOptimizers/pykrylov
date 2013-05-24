"""
Copyright (C) 2010 David Fong and Michael Saunders
Distributed under the same license as Scipy

LSMR uses an iterative method.

07 Jun 2010: Documentation updated
03 Jun 2010: First release version in Python

David Chin-lung Fong            clfong@stanford.edu
Institute for Computational and Mathematical Engineering
Stanford University

Michael Saunders                saunders@stanford.edu
Systems Optimization Laboratory
Dept of MS&E, Stanford University.

"""

__docformat__ = 'restructuredtext'

from pykrylov.generic import KrylovMethod

from numpy import zeros, dot
from numpy.linalg import norm
from math import sqrt

class LSMRFramework(KrylovMethod):

    def __init__(self, A, **kwargs):

        # Initialize.
        KrylovMethod.__init__(self, A, **kwargs)
        self.name = 'Least-Squares Minimum Residual'
        self.acronym = 'LSMR'
        self.prefix = self.acronym + ': '

        self.msg=('The exact solution is  x = 0                              ',
                  'Ax - b is small enough, given atol, btol                  ',
                  'The least-squares solution is good enough, given atol     ',
                  'The estimate of cond(Abar) has exceeded conlim            ',
                  'Ax - b is small enough for this machine                   ',
                  'The least-squares solution is good enough for this machine',
                  'Cond(Abar) seems to be too large for this machine         ',
                  'The iteration limit has been reached                      ',
                  'The truncated direct error is small enough, given etol    ')

        self.A = A
        self.x = None ; self.var = None

        self.itn = 0; self.istop = 0
        self.Anorm = 0.; self.Acond = 0. ; self.Arnorm = 0.
        self.xnorm = 0.;
        self.r1norm = 0.; self.r2norm = 0.
        self.optimal = False
        self.resids = []             # Least-squares objective function values.
        self.normal_eqns_resids = [] # Residuals of normal equations.
        self.norms = []              # Squared energy norm of iterates.
        self.dir_errors_window = []  # Direct error estimates.
        self.iterates = []
        return


    def solve(self, b, damp=0.0, atol=1e-9, btol=1e-9, conlim=1e8,
              M=None, N=None, itnlim=None, show=False, **kwargs):
        """
        Iterative solver for least-squares problems.

        lsmr solves the system of linear equations A*x=b. If the system
        is inconsistent, it solves the least-squares problem min ||b - Ax||_2.
        A is a rectangular matrix of dimension m-by-n, where all cases are
        allowed: m=n, m>n, or m<n. B is a vector of length m.
        The matrix A may be dense or sparse (usually sparse).

        :parameters:

            :A: LinearOperator
                matrix A in the linear system
            :b: (m,) ndarray
                vector b in the linear system
            :damp: float
                Damping factor for regularized least-squares. lsmr solves
                the regularized least-squares problem

                min ||b - Ax|| + damp * ||x||

                where damp is a scalar.  If damp is None or 0, the system
                is solved without regularization.
            :atol:
            :btol: float
                Stopping tolerances. lsmr continues iterations until a certain
                backward error estimate is smaller than some quantity depending
                on atol and btol.  Let r = b - A*x be the residual vector for
                the current approximate solution x.  If A*x = b seems to be
                consistent, lsmr terminates when norm(r) <= atol*norm(A)*norm(x)
                + btol*norm(b).  Otherwise, lsmr terminates when norm(A^{T}*r)
                <= atol*norm(A)*norm(r).  If both tolerances are 1.0e-6 (say),
                the final norm(r) should be accurate to about 6 digits. (The final
                x will usually have fewer correct digits, depending on cond(A)
                and the size of LAMBDA.)  If atol or btol is None, a default
                value of 1.0e-6 will be used.  Ideally, they should be estimates
                of the relative error in the entries of A and B respectively.
                For example, if the entries of A have 7 correct digits, set atol
                = 1e-7. This prevents the algorithm from doing unnecessary work
                beyond the uncertainty of the input data.
            :etol: float
                stopping tolerance based on direct error (default 1.0e-6).
            :conlim: float
                lsmr terminates if an estimate of cond(A) exceeds conlim.
                For compatible systems Ax = b, conlim could be as large as 1.0e+12
                (say).  For least-squares problems, conlim should be less than
                1.0e+8. If conlim is None, the default value is CONLIM = 1e+8.
                Maximum precision can be obtained by setting atol = btol =
                conlim = 0, but the number of iterations may then be excessive.
            :itnlim: int
                lsmr terminates if the number of iterations reaches itnlim.
                The default is itnlim = min(m,n).  For ill-conditioned systems,
                a larger value of itnlim may be needed.
            :show: bool
                print iterations logs if show=True
            :store_resids: bool
                Store full residual norm history. Default: False
            :window: int
                Number of consecutive iterations over which the director error
                should be measured (default: 5).

        :returns:

            :x: ndarray of float
                least-square solution returned
            :istop: int
                istop gives the reason for stopping.
                istop   = 0 means x=0 is a solution.
                        = 1 means x is an approximate solution to A*x = B,
                            according to atol and btol.
                        = 2 means x approximately solves the least-squares problem
                            according to atol.
                        = 3 means COND(A) seems to be greater than CONLIM.
                        = 4 is the same as 1 with atol = btol = eps (machine
                            precision)
                        = 5 is the same as 2 with atol = eps.
                        = 6 is the same as 3 with CONLIM = 1/eps.
                        = 7 means ITN reached itnlim before the other stopping
                            conditions were satisfied.
                        = 8 means that the truncated direct error estimate is
                            sufficiently small.
            :itn: int
                number of iterations used
            :normr: float
                norm(b-A*x)
            :normar: float
                norm(A^T *(b-A*x))
            :norma: float
                norm(A)
            :conda: float
                condition number of A
            :normx: float
                norm(x)

        Reference

        http://arxiv.org/abs/1006.0758
        D. C.-L. Fong and M. A. Saunders
        LSMR: An iterative algorithm for least-square problems
        Draft of 01 Jun 2010, submitted to SISC.

        """

        etol = kwargs.get('etol', 1.0e-6)
        store_resids = kwargs.get('store_resids', False)
        store_iterates = kwargs.get('store_iterates', False)
        window = kwargs.get('window', 5)

        self.resids = []             # Least-squares objective function values.
        self.normal_eqns_resids = [] # Residuals of normal equations.
        self.norms = []              # Squared energy norm of iterates.
        self.dir_errors_window = []  # Direct error estimates.
        self.iterates = []

        A = self.A
        b = b.squeeze()
        msg = self.msg

        hdg1 = '   itn      x(1)       norm r    norm A''r'
        hdg2 = ' compatible   LS      norm A   cond A'
        pfreq  = 20   # print frequency (for repeating the heading)
        pcount = 0    # print counter

        m, n = A.shape

        # stores the num of singular values
        minDim = min([m, n])

        if itnlim is None: itnlim = minDim

        if N is not None: damp = 1.0

        if show:
            print ' '
            print 'LSMR            Least-squares solution of  Ax = b'
            str1 = 'The matrix A has %8g rows  and %8g cols' % (m, n)
            str2 = 'damp = %20.14e' % (damp)
            str3 = 'atol = %8.2e                 conlim = %8.2e'%( atol, conlim)
            str4 = 'btol = %8.2e               itnlim = %8g'  %( btol, itnlim)
            print str1
            print str2
            print str3
            print str4

        # Initialize the Golub-Kahan bidiagonalization process.

        Mu = b.copy()
        if M is not None:
            u = M(Mu)
        else:
            u = Mu
        beta = sqrt(dot(u,Mu))  # norm(u)

        v = zeros(n)
        alpha = 0

        if beta > 0:
            u /= beta
            if M is not None: Mu /= beta

            Nv = A.T * u
            if N is not None:
                v = N(Nv)
            else:
                v = Nv
            alpha = sqrt(dot(v,Nv))  # norm(v)

        if alpha > 0:
            v /= alpha
            if N is not None: Nv /= alpha

        # Initialize variables for 1st iteration.

        itn      = 0
        zetabar  = alpha*beta
        alphabar = alpha
        rho      = 1
        rhobar   = 1
        cbar     = 1
        sbar     = 0

        h    = v.copy()
        hbar = zeros(n)
        x    = zeros(n)

        if store_iterates:
            self.iterates.append(x.copy())

        # Initialize variables for estimation of ||r||.

        betadd      = beta
        betad       = 0
        rhodold     = 1
        tautildeold = 0
        thetatilde  = 0
        zeta        = 0
        d           = 0

        # Initialize variables for estimation of ||A|| and cond(A)

        normA2  = alpha*alpha
        maxrbar = 0
        minrbar = 1e+100
        normA   = sqrt(normA2)
        condA   = 1
        normx   = 0
        xNrgNorm2 = 0  # norm(x)^2 in the appropriate energy norm.
        dErr = zeros(window)     # Truncated direct error terms.
        trncDirErr = 0           # Truncated direct error.

        # Items for use in stopping rules.
        normb  = beta
        istop  = 0
        ctol   = 0
        if conlim > 0: ctol = 1/conlim
        normr  = beta

        # Reverse the order here from the original matlab code because
        # there was an error on return when arnorm==0
        normar = alpha * beta
        if normar == 0:
            if show:
                print msg[0]
            return x, istop, itn, normr, normar, normA, condA, normx

        if show:
            print ' '
            print hdg1, hdg2
            test1  = 1;             test2  = alpha / beta
            str1   = '%6g %12.5e'    %(    itn,   x[0] )
            str2   = ' %10.3e %10.3e'%(  normr, normar )
            str3   = '  %8.1e %8.1e' %(  test1,  test2 )
            print ''.join([str1, str2, str3])

        if store_resids:
            self.resids.append(normr)
            self.normal_eqns_resids.append(normar)

        # Main iteration loop.
        while itn < itnlim:
            itn = itn + 1

            # Perform the next step of the bidiagonalization to obtain the
            # next  beta, u, alpha, v.  These satisfy the relations
            #         beta*M*u  =  A*v   -  alpha*M*u,
            #        alpha*C*v  =  A'*u  -  beta*C*v.

            Mu = A * v - alpha * Mu
            if M is not None:
                u = M(Mu)
            else:
                u = Mu
            beta = sqrt(dot(u,Mu))  # norm(u)

            if beta > 0:
                u /= beta
                if M is not None: Mu /= beta

                Nv = A.T * u - beta * Nv
                if N is not None:
                    v = N(Nv)
                else:
                    v = Nv

                alpha  = sqrt(dot(v,Nv))  # norm(v)

                if alpha > 0:
                    v /= alpha
                    if N is not None: Nv /= alpha

            # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

            # Construct rotation Qhat_{k,2k+1}.

            chat, shat, alphahat = symOrtho(alphabar, damp)

            # Use a plane rotation (Q_i) to turn B_i to R_i

            rhoold   = rho
            c, s, rho = symOrtho(alphahat, beta)
            thetanew = s*alpha
            alphabar = c*alpha

            # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

            rhobarold = rhobar
            zetaold   = zeta
            thetabar  = sbar*rho
            rhotemp   = cbar*rho
            cbar, sbar, rhobar = symOrtho(cbar*rho, thetanew)
            zeta      =   cbar*zetabar
            zetabar   = - sbar*zetabar

            # Update h, h_hat, x.

            hbar       = h - (thetabar*rho/(rhoold*rhobarold))*hbar
            x          = x + (zeta/(rho*rhobar))*hbar
            h          = v - (thetanew/rho)*h

            if store_iterates:
                self.iterates.append(x.copy())

            xNrgNorm2 +=  zeta * zeta
            dErr[itn % window] = zeta
            if itn > window:
                trncDirErr = norm(dErr)
                xNrgNorm = sqrt(xNrgNorm2)
                self.dir_errors_window.append(trncDirErr / xNrgNorm)
                if trncDirErr < etol * xNrgNorm:
                    istop = 8

            # Estimate of ||r||.

            # Apply rotation Qhat_{k,2k+1}.
            betaacute =   chat* betadd
            betacheck = - shat* betadd

            # Apply rotation Q_{k,k+1}.
            betahat   =   c*betaacute
            betadd    = - s*betaacute

            # Apply rotation Qtilde_{k-1}.
            # betad = betad_{k-1} here.

            thetatildeold = thetatilde
            ctildeold, stildeold, rhotildeold = symOrtho(rhodold, thetabar)
            thetatilde    = stildeold* rhobar
            rhodold       =   ctildeold* rhobar
            betad         = - stildeold*betad + ctildeold*betahat

            # betad   = betad_k here.
            # rhodold = rhod_k  here.

            tautildeold   = (zetaold - thetatildeold*tautildeold)/rhotildeold
            taud          = (zeta - thetatilde*tautildeold)/rhodold
            d             = d + betacheck*betacheck
            normr         = sqrt(d + (betad - taud)**2 + betadd*betadd)

            # Estimate ||A||.
            normA2        = normA2 + beta*beta
            normA         = sqrt(normA2)
            normA2        = normA2 + alpha*alpha

            # Estimate cond(A).
            maxrbar       = max(maxrbar,rhobarold)
            if itn>1:
              minrbar     = min(minrbar,rhobarold)
            condA         = max(maxrbar,rhotemp)/min(minrbar,rhotemp)

            # Test for convergence.

            # Compute norms for convergence testing.
            normar  = abs(zetabar)
            normx   = norm(x)

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.

            test1   = normr /normb
            test2   = normar/(normA*normr)
            test3   =      1/condA
            t1      =  test1/(1 + normA*normx/normb)
            rtol    = btol + atol*normA*normx/normb

            if store_resids:
                self.norms.append(xNrgNorm2)
                self.resids.append(normr)
                self.normal_eqns_resids.append(normar)

            # The following tests guard against extremely small values of
            # atol, btol or ctol.  (The user may have set any or all of
            # the parameters atol, btol, conlim  to 0.)
            # The effect is equivalent to the normAl tests using
            # atol = eps,  btol = eps,  conlim = 1/eps.

            if itn >= itnlim:   istop = 7
            if 1 + test3  <= 1: istop = 6
            if 1 + test2  <= 1: istop = 5
            if 1 + t1     <= 1: istop = 4

            # Allow for tolerances set by the user.

            if  test3 <= ctol:  istop = 3
            if  test2 <= atol:  istop = 2
            if  test1 <= rtol:  istop = 1

            # See if it is time to print something.

            if show:
                prnt = 0
                if n     <= 40       : prnt = 1
                if itn   <= 10       : prnt = 1
                if itn   >= itnlim-10: prnt = 1
                if itn % 10 == 0     : prnt = 1
                if test3 <= 1.1*ctol : prnt = 1
                if test2 <= 1.1*atol : prnt = 1
                if test1 <= 1.1*rtol : prnt = 1
                if istop !=  0       : prnt = 1

                if prnt:
                    if pcount >= pfreq:
                        pcount = 0
                        print ' '
                        print hdg1, hdg2
                    pcount = pcount + 1
                    str1   = '%6g %12.5e'    %(    itn,   x[0] )
                    str2   = ' %10.3e %10.3e'%(  normr, normar )
                    str3   = '  %8.1e %8.1e' %(  test1,  test2 )
                    str4   = ' %8.1e %8.1e'  %(  normA,  condA )
                    print ''.join([str1, str2, str3, str4])

            if istop > 0: break

        # Print the stopping condition.

        if show:
            print ' '
            print 'LSMR finished'
            print msg[istop]
            str1    = 'istop =%8g    normr =%8.1e'      %( istop, normr )
            str2    = '    normA =%8.1e    normAr =%8.1e' %( normA, normar)
            str3    = 'itn   =%8g    condA =%8.1e'      %( itn  , condA )
            str4    = '    normx =%8.1e'                %( normx)
            print str1, str2
            print str3, str4
            print 'Estimated energy norm of x: %7.1e' % sqrt(xNrgNorm2)

        self.x = x
        return x, istop, itn, normr, normar, normA, condA, normx


def sign(a):
    if a < 0: return -1
    return 1


def symOrtho(a,b):
    """
    A stable implementation of Givens rotation according to
    S.-C. Choi, "Iterative Methods for Singular Linear Equations
      and Least-Squares Problems", Dissertation,
      http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    if b==0: return sign(a), 0, abs(a)
    elif a==0: return 0, sign(b), abs(b)
    elif abs(b)>abs(a):
        tau = a / b
        s = sign(b) / sqrt(1+tau*tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = sign(a) / sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r
