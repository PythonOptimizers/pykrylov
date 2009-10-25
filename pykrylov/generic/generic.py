
__docformat__ = 'restructuredtext'

class KrylovMethod:
    """
    A general template for implementing iterative Krylov methods. This module
    defines the `KrylovMethod` generic class. Other modules subclass
    `KrylovMethod` to implement specific algorithms.

    :parameters:

        :matvec:  an operator describing the coefficient matrix `A`.
                  `y = matvec(x)` must return the matrix-vector product
                  `y = Ax` for any given vector `x`.

    :keywords:

        :atol:    absolute stopping tolerance. Default: 1.0e-8.

        :rtol:    relative stopping tolerance. Default: 1.0e-6.

        :verbose: verbosity flag. Default: False.

        :precon:  optional preconditioner. If not `None`, `y = precon(x)`
                  returns the vector `y` solution of the linear system
                  `M y = x`.


    For general references on Krylov methods, see [Demmel]_, [Greenbaum]_,
    [Kelley]_, [Saad]_ and [Templates]_.

    References:

    .. [Demmel] J. W. Demmel, *Applied Numerical Linear Algebra*, SIAM,
                Philadelphia, 1997.

    .. [Greenbaum] A. Greenbaum, *Iterative Methods for Solving Linear Systems*,
                   number 17 in *Frontiers in Applied Mathematics*, SIAM,
                   Philadelphia, 1997.

    .. [Kelley] C. T. Kelley, *Iterative Methods for Linear and Nonlinear
                Equations*, number 16 in *Frontiers in Applied Mathematics*,
                SIAM, Philadelphia, 1995.

    .. [Saad] Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed.,
              SIAM, Philadelphia, 2003.

    .. [Templates] R. Barrett, M. Berry, T. F. Chan, J. Demmel, J. M. Donato,
                   J. Dongarra, V. Eijkhout, R. Pozo, C. Romine and
                   H. Van der Vorst, *Templates for the Solution of Linear
                   Systems: Building Blocks for Iterative Methods*, SIAM,
                   Philadelphia, 1993.
    """

    def __init__(self, matvec, **kwargs):

        self.prefix = 'Generic: '
        self.name   = 'Generic Krylov Method (must be subclassed)'

        # Mandatory arguments
        self.matvec = matvec

        # Optional keyword arguments
        self.verbose = kwargs.get('verbose', False)
        self.abstol = kwargs.get('abstol', 1.0e-8)
        self.reltol = kwargs.get('reltol', 1.0e-6)
        self.precon = kwargs.get('precon', None)
        #self.matvec_max = kwargs.get('matvec_max', None)
        self.outputStream = kwargs.get('outputStream', None)

        self.residNorm  = None
        self.residNorm0 = None

        self.nMatvec = 0
        self.nIter = 0
        self.converged = False
        self.bestSolution = None

    def _write(self, msg):
        if self.outputStream is not None:
            self.outputStream.write(self.prefix + msg)
        return None

    def solve(self, rhs, **kwargs):
        """
        This is the :meth:`solve` method of the abstract KrylovMethod class.
        The class must be specialized and this method overridden.
        """
        raise NotImplementedError, 'This method must be subclassed'
