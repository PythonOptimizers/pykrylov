"""
Solve the linear system

  A x = b

or the least-squares problem

  minimize ||Ax-b||

using Minres.  This is a line-by-line translation from Matlab code
available at http://www.stanford.edu/group/SOL/software/minres.htm.

.. moduleauthor:: D. Orban <dominique.orban@gerad.ca>
"""

from numpy import zeros, dot, empty
from types import FunctionType
from math import sqrt

class Minres:
    """
    `K = Minres(A) ; K.solve(b)`

    This class implements the Minres iterative solver of Paige and Saunders.
    Minres solves the system of linear equations Ax = b
    or the least squares problem           min ||Ax - b||_2^2,
    where A is a symmetric matrix (possibly indefinite or singular)
    and b is a given vector.
   
    A may be given explicitly as a matrix or be a function such that

        `A(x, y)`

    stores in y the product Ax for any given vector x.
    If A is an instance of some matrix class, it should have a 'matvec' method
    such that A.matvec(x, y) has the behaviour described above.
 
    Optional keyword arguments are:

        precon    optional preconditioner, given as an operator        (None)
        shift     optional shift value                                 (0.0)
        show      display information along the iterations             (True)
        check     perform some argument checks                         (True)
        itnlim    maximum number of iterations                         (5n)
        rtol      relative stopping tolerance                          (1.0e-12)

    If precon is given, it must define a positive-definite preconditioner
    M = C C'. The precon operator must be such that

        `x = precon(y)`

    returns the solution x to the linear system M x = y, for any given y.
 
    If shift != 0, minres really solves (A - shift*I)x = b
    or the corresponding least squares problem if shift is an eigenvalue of A.

    The return values (as returned by the Matlab version of Minres) are stored
    in the members

        `x`, `istop`, `itn`, `rnorm`, `Arnorm`, `Anorm`, `Acond`, `ynorm`

    of the class, after completion of :meth:`solve`.

    Python version: Dominique Orban, Ecole Polytechnique de Montreal, 2008,
    translated and adapted from the Matlab version of Minres, written by

        Michael Saunders, SOL, Stanford University
        Sou Cheng Choi,  SCCM, Stanford University

    See also http://www.stanford.edu/group/SOL/software/minres.html
    """

#    02 Sep 2003: Date of Fortran 77 version, based on 
#                 C. C. Paige and M. A. Saunders (1975),
#                 Solution of sparse indefinite systems of linear equations,
#                 SIAM J. Numer. Anal. 12(4), pp. 617-629.
# 
#    02 Sep 2003: ||Ar|| now estimated as Arnorm.
#    17 Oct 2003: f77 version converted to MATLAB.
#    11 Jan 2008: MATLAB version converted to Python.

    def __init__(self, A, **kwargs):

        self.A = A
        self.x = None

        # Read keyword arguments
        self.precon = kwargs.get('precon', None)
        self.shift  = kwargs.get('shift',  0.0)
        self.show   = kwargs.get('show',   True)
        self.check  = kwargs.get('check',  True)

        #  Initialize
        self.first = 'Enter minres.   '
        self.last  = 'Exit  minres.   '
        self.space = ' '
        self.msg = [' beta2 = 0.  If M = I, b and x are eigenvectors    ',  #-1
                    ' beta1 = 0.  The exact solution is  x = 0          ',  # 0
                    ' A solution to Ax = b was found, given rtol        ',  # 1
                    ' A least-squares solution was found, given rtol    ',  # 2
                    ' Reasonable accuracy achieved, given eps           ',  # 3
                    ' x has converged to an eigenvector                 ',  # 4
                    ' acond has exceeded 0.1/eps                        ',  # 5
                    ' The iteration limit was reached                   ',  # 6
                    ' Aname  does not define a symmetric matrix         ',  # 7
                    ' Mname  does not define a symmetric matrix         ',  # 8
                    ' Mname  does not define a pos-def preconditioner   ' ] # 9
 
        #---------------------------------------------------------------------
        # See if A is explicit or an operator
        #---------------------------------------------------------------------
        self.explicitA = hasattr(A,'matvec') #not isinstance(self.A, FunctionType)

        if self.explicitA:            # assume Aname is an explicit matrix A.
            if hasattr(A,'nnz'):
                nnzA   = self.A.nnz
                print 'A is an explicit matrix with %d nonzeros.' % nnzA
            else:
                print 'A is an explicit matrix'
        else:
            print 'A is an operator.'

        self.eps = self._Epsilon()


    def _Epsilon(self):
        """
        Return approximate value of machine epsilon
        """
        one = 1.0
        eps = 1.0
        while (one + eps) > one:
            eps = eps / 2.0
        return eps*2.0

    def applyA(self, x, y):
        """
        Given x, compute the matrix-vector product y = Ax
        """
        if self.explicitA:
            self.A.matvec(x,y)
        else:
            self.A(x,y)
        return

    def solve(self, b, **kwargs):

        n = b.shape[0]
        x = zeros(n)

        itnlim = kwargs.get('itnlim', 5*n)
        rtol   = kwargs.get('rtol',   1.0e-12)

        # Transfer some pointers for readability
        shift = self.shift
        eps = self.eps

        if self.show:
            print self.space
            print self.first + 'Solution of symmetric Ax = b'
            print 'n      =  %3d     precon =  %4s           shift  =  %23.14e'\
                % (n, str(self.precon != None), self.shift)
            print 'itnlim =  %3d     rtol   =  %11.2e\n' % (itnlim, rtol)

        istop = 0;   itn = 0;     Anorm = 0.0;    Acond = 0.0;
        rnorm = 0.0; ynorm = 0.0; done  = False;

        #------------------------------------------------------------------
        # Set up y and v for the first Lanczos vector v1.
        # y  =  beta1 P' v1,  where  P = C**(-1).
        # v is really P' v1.
        #------------------------------------------------------------------
        r1 = b
        if self.precon is not None:
            y = self.precon(b)
        else:
            y = b.copy()
        beta1 = dot(b,y)

        #  Test for an indefinite preconditioner.
        #  If b = 0 exactly, stop with x = 0.
        if beta1 < 0:
            istop = 8
            self.show = True
            done = True

        if beta1 == 0.0:
            self.show = True
            done = True

        if beta1 > 0:
            beta1 = sqrt(beta1);       # Normalize y to get v1 later.

        # See if A is symmetric.
        if self.check:
            w = empty(n)
            r2 = empty(n)
            self.applyA(y,w)
            self.applyA(w,r2)
            s    = dot(w,w)
            t    = dot(y,r2)
            print 's = ', s, ', t = ', t
            z    = abs(s - t)
            epsa = (s + eps) * eps**(1.0/3)
            print 'z = ', z, ', epsa = ', epsa
            if z > epsa:
                istop = 6
                done  = True
                self.show = True

        # See if preconditioner is symmetric.
        if self.check and self.precon is not None:
            r2 = self.precon(y)
            s    = dot(y,y)
            t    = dot(r1,r2)
            z    = abs(s - t)
            epsa = (s + eps) * eps**(1.0/3)
            if z > epsa:
                istop = 7
                self.show = True
                done = True

        # -------------------------------------------------------------------
        # Initialize other quantities.
        # ------------------------------------------------------------------
        oldb   = 0.0;     beta   = beta1;   dbar   = 0.0;     epsln  = 0.0
        qrnorm = beta1;   phibar = beta1;   rhs1   = beta1;   Arnorm = 0.0
        rhs2   = 0.0;     tnorm2 = 0.0;     ynorm2 = 0.0
        cs     = -1.0;    sn     = 0.0
        w  = zeros(n)
        w2 = zeros(n)
        r2 = r1.copy()

        if self.show:
            print ' '*2
            head1 = '   Itn     x[0]     Compatible    LS'
            head2 = '       norm(A)  cond(A) gbar/|A|'   ###### Check gbar
            print head1 + head2

        # ---------------------------------------------------------------------
        # Main iteration loop.
        # --------------------------------------------------------------------
        print 'done = ', done
        if not done:                          # k = itn = 1 first time through
            while itn < itnlim:
                itn    = itn  +  1

                # -------------------------------------------------------------
                # Obtain quantities for the next Lanczos vector vk+1, k=1,2,...
                # The general iteration is similar to the case k=1 with v0 = 0:
                #
                #   p1      = Operator * v1  -  beta1 * v0,
                #   alpha1  = v1'p1,
                #   q2      = p2  -  alpha1 * v1,
                #   beta2^2 = q2'q2,
                #   v2      = (1/beta2) q2.
                #
                # Again, y = betak P vk,  where  P = C**(-1).
                # .... more description needed.
                # -------------------------------------------------------------
                s = 1.0/beta                # Normalize previous vector (in y).
                v = s*y                     # v = vk if P = I

                self.applyA(v,y)
                y      = (- shift)*v + y

                if itn >= 2:
                    y = y - (beta/oldb)*r1

                alfa = dot(v,y)           # alphak
                y    = (- alfa/beta)*r2 + y
                r1   = r2.copy()
                r2   = y.copy()
                if self.precon is not None: y = self.precon(r2)
                oldb   = beta               # oldb = betak
                beta   = dot(r2,y)          # beta = betak+1^2
                if beta < 0:
                    istop = 6
                    break
                beta   = sqrt(beta)
                tnorm2 = tnorm2 + alfa**2 + oldb**2 + beta**2

                if itn==1:                  # Initialize a few things.
                    if beta/beta1 <= 10*eps:  # beta2 = 0 or ~ 0.
                        istop = -1            # Terminate later.

                    # tnorm2 = alfa**2  ??
                    gmax   = abs(alfa)      # alpha1
                    gmin   = gmax             # alpha1

                # Apply previous rotation Qk-1 to get
                #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
                #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

                oldeps = epsln
                delta  = cs * dbar  +  sn * alfa  # delta1 = 0         deltak
                gbar   = sn * dbar  -  cs * alfa  # gbar 1 = alfa1     gbar k

                # Note: There is severe cancellation in the computation of gbar
                #print ' sn = %21.15e\n dbar = %21.15e\n cs = %21.15e\n alfa = %21.15e\n sn*dbar-cs*alfa = %21.15e\n gbar =%21.15e' % (sn, dbar, cs, alfa, sn*dbar-cs*alfa, gbar)

                epsln  =               sn * beta  # epsln2 = 0         epslnk+1
                dbar   =            -  cs * beta  # dbar 2 = beta2     dbar k+1
                root   = self.normof2(gbar, dbar)
                Arnorm = phibar * root
                
                # Compute the next plane rotation Qk

                gamma  = self.normof2(gbar, beta)       # gammak
                gamma  = max(gamma, eps)
                cs     = gbar / gamma             # ck
                sn     = beta / gamma             # sk
                phi    = cs * phibar              # phik
                phibar = sn * phibar              # phibark+1

                # Update  x.

                denom = 1.0/gamma
                w1    = w2.copy()
                w2    = w.copy()
                w     = (v - oldeps*w1 - delta*w2) * denom
                x     = x  +  phi*w

                # Go round again.

                gmax   = max(gmax, gamma)
                gmin   = min(gmin, gamma)
                z      = rhs1 / gamma
                ynorm2 = z**2  +  ynorm2
                rhs1   = rhs2 -  delta*z
                rhs2   =      -  epsln*z

                # Estimate various norms and test for convergence.

                Anorm  = sqrt(tnorm2)
                ynorm  = sqrt(ynorm2)
                epsa   = Anorm * eps
                epsx   = Anorm * ynorm * eps
                epsr   = Anorm * ynorm * rtol
                diag   = gbar
                if diag==0: diag = epsa

                qrnorm = phibar
                rnorm  = qrnorm
                test1  = rnorm / (Anorm*ynorm)     #  ||r|| / (||A|| ||x||)
                test2  = root  /  Anorm            # ||Ar|| / (||A|| ||r||)

                # Estimate  cond(A).
                # In this version we look at the diagonals of  R  in the
                # factorization of the lower Hessenberg matrix,  Q * H = R,
                # where H is the tridiagonal matrix from Lanczos with one
                # extra row, beta(k+1) e_k^T.

                Acond  = gmax/gmin

                # See if any of the stopping criteria are satisfied.
                # In rare cases istop is already -1 from above (Abar = const*I)

                if istop==0:
                    t1 = 1 + test1      # These tests work if rtol < eps
                    t2 = 1 + test2
                    if t2 <= 1: istop = 2
                    if t1 <= 1: istop = 1
      
                    if itn >= itnlim: istop = 6
                    if Acond >= 0.1/eps: istop = 4
                    if epsx >= beta1: istop = 3
                    # if rnorm <= epsx: istop = 2
                    # if rnorm <= epsr: istop = 1
                    if test2 <= rtol: istop = 2
                    if test1 <= rtol: istop = 1

                # See if it is time to print something.

                prnt   = False
                if n <= 40: prnt = True
                if itn <= 10: prnt = True
                if itn >= itnlim-10: prnt = True
                if (itn % 10)==0: prnt = True
                if qrnorm <= 10*epsx: prnt = True
                if qrnorm <= 10*epsr: prnt = True
                if Acond <= 1e-2/eps: prnt = True
                if istop !=  0: prnt = True

                if self.show and prnt:
                    str1 = '%6g %12.5e %10.3e' % (itn, x[0], test1)
                    str2 = ' %10.3e' % test2
                    str3 = ' %8.1e %8.1e %8.1e' % (Anorm, Acond, gbar/Anorm)
                    print str1 + str2 + str3

                if istop > 0: break

                if (itn % 10)==0: print ' '

        # Display final status.

        if self.show:
            print self.space
            last = self.last
            print last+' istop   =  %3g               itn   =%5g' % (istop,itn)
            print last+' Anorm   =  %12.4e      Acond =  %12.4e' %(Anorm,Acond)
            print last+' rnorm   =  %12.4e      ynorm =  %12.4e' % (rnorm,ynorm)
            print last+' Arnorm  =  %12.4e' % Arnorm
            print last+self.msg[istop+1]

        self.x = x
        self.istop = istop
        self.itn = itn
        self.rnorm = rnorm
        self.Arnorm = Arnorm
        self.Anorm = Anorm
        self.Acond = Acond
        self.ynorm = ynorm

        return
    # -----------------------------------------------------------------------
    # End function minres
    # -----------------------------------------------------------------------

    def normof2(self, x,y):
        return sqrt(x**2 + y**2)
