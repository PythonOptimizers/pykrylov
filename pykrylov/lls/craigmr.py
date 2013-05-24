__docformat__ = 'restructuredtext'

# TODO:
#  []  Compute least-squares objective

from pykrylov.generic import KrylovMethod

from numpy import zeros, dot
from numpy.linalg import norm
from math import sqrt


class CRAIGMRFramework(KrylovMethod):

    def __init__(self, A, **kwargs):

        # Initialize.
        KrylovMethod.__init__(self, A, **kwargs)
        self.name = 'Least-Norm Minimum Residual'
        self.acronym = 'CRAIG-MR'
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
        self.init_data()

    def init_data(self):

        self.x = None ; self.var = None
        self.itn = 0; self.istop = 0
        self.Anorm = 0.; self.Acond = 0. ; self.Arnorm = 0.
        self.xnorm = 0.;
        self.r1norm = 0.; self.r2norm = 0.
        self.optimal = False
        self.resids = []             # Least-squares objective function values.
        self.normal_eqns_resids = [] # Residuals of normal equations.
        self.norms = []              # Squared energy norm of iterates.
        self.dir_errors_window = []
        self.iterates = []
        return

    def solve(self, b, damp=0.0, atol=1e-9, btol=1e-9, conlim=1e8,
              M=None, N=None, itnlim=None, show=False, **kwargs):

        etol = kwargs.get('etol', 1.0e-6)
        store_resids = kwargs.get('store_resids', False)
        store_iterates = kwargs.get('store_iterates', False)
        window = kwargs.get('window', 5)

        # Reinitialize internal data for multiple solves.
        self.init_data()

        A = self.A
        b = b.squeeze()
        msg = self.msg

        m, n = A.shape

        # stores the num of singular values
        minDim = min([m, n])

        if itnlim is None: itnlim = minDim

        if N is not None: damp = 1.0

        # Initialize the Golub-Kahan bidiagonalization process.

        Mu = b.copy()  # Don't want to change input vector.
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

        itn       = 0
        delta     = 1.
        alpha_hat = sqrt(alpha**2 + 1)
        c         = alpha / alpha_hat
        s         = 1.    / alpha_hat
        zeta_hat  = beta
        alpha_tilde = alpha_hat
        theta     = 0.
        d         = u / alpha_hat
        dbar      = zeros(m)
        x         = zeros(m)

        if store_iterates:
            self.iterates.append(x.copy())

        xNrgNorm2 = 0.  # norm(x)^2 in the appropriate energy norm.
        dErr = zeros(window)     # Truncated direct error terms.
        trncDirErr = 0           # Truncated direct error.

        # Items for use in stopping rules.
        istop = 0

        if store_resids:
            self.norms.append(xNrgNorm2)

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

            # Continue previous rotation of type I.
            beta_hat = c * beta
            gamma    = s * beta

            # Compute rotation of type II.
            delta    = sqrt(gamma**2 + 1)
            c_bar    = -1. / delta
            s_bar    = gamma / delta

            # Compute new rotation of type I.
            alpha_hat = sqrt(alpha**2 + delta**2)
            c         = alpha / alpha_hat
            s         = delta / alpha_hat

            # Compute rotation of type III.
            rho   = sqrt(alpha_tilde**2 + beta_hat**2)
            c_hat = alpha_tilde / rho
            s_hat = beta_hat    / rho

            # Update dbar before computing new theta.
            dbar = (d - theta * dbar) / rho

            theta = s_hat * alpha_hat
            alpha_tilde = -c_hat * alpha_hat

            # Updates.
            zeta     = c_hat * zeta_hat
            zeta_hat = s_hat * zeta_hat
            xNrgNorm2 += zeta * zeta
            print itn, xNrgNorm2
            d = (u - beta_hat * d) / alpha_hat
            x += zeta * dbar

            if store_iterates:
                self.iterates.append(x.copy())

            if store_resids:
                self.norms.append(xNrgNorm2)
                self.normal_eqns_resids.append(abs(zeta))

            # See if it's time to stop.
            dErr[itn % window] = zeta
            if itn > window:
                trncDirErr = norm(dErr)
                xNrgNorm = sqrt(xNrgNorm2)
                self.dir_errors_window.append(trncDirErr / xNrgNorm)
                if trncDirErr < etol * xNrgNorm:
                    istop = 8

            if itn >= itnlim:   istop = 7

            if istop > 0: break

        if show:
            print ' '
            print 'CRAIG-MR finished'
            print self.msg[istop]
            print ' '
            #str1 = 'istop =%8g   r1norm =%8.1e'   % (istop, sqrt(r1norm))
            #str2 = 'Anorm =%8.1e   Arnorm =%8.1e' % (Anorm, Arnorm)
            #str3 = 'itn   =%8g   r2norm =%8.1e'   % ( itn, sqrt(r2norm))
            #str4 = 'Acond =%8.1e   xnorm  =%8.1e' % (Acond, xnorm )
            #str5 = '                  bnorm  =%8.1e'    % bnorm
            str6 = 'xNrgNorm2 = %7.1e   trnDirErr = %7.1e' % \
                    (xNrgNorm2, trncDirErr)
            #print str1 #+ '   ' + str2
            #print str3 #+ '   ' + str4
            #print str5
            print str6
            print ' '

        if istop == 0: self.status = 'solution is zero'
        if istop in [1,2,4,5]: self.status = 'residual small'
        if istop in [3,6]: self.status = 'ill-conditioned operator'
        if istop == 7: self.status = 'max iterations'
        if istop == 8: self.status = 'direct error small'
        self.optimal = istop in [1,2,4,5,8]
        self.x = self.bestSolution = x
        self.istop = istop
        self.itn = itn
        self.nMatvec = 2*itn
        #self.r1norm = sqrt(r1norm)
        #self.r2norm = sqrt(r2norm)
        #self.residNorm = r2norm
        #self.Anorm = Anorm
        #self.Acond = Acond
        #self.Arnorm = Arnorm
        #self.xnorm = xnorm

        return
