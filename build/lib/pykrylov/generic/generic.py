"""
A general template for implementing iterative Krylov methods. This module
defines the `KrylovMethod` generic class. Other modules subclass `KrylovMethod`
to implement specific algorithms.
"""

__docformat__ = 'restructuredtext'

class KrylovMethod:

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
        self.matvec_max = kwargs.get('matvec_max', None)
        self.outputStream = kwargs.get('outputStream', None)

        self.residNorm  = None
        self.residNorm0 = None

        self.nMatvec = 0
        self.nIter = 0
        self.converged = False
        self.bestSolution = None

    def _write(self, msg):
       self.outputStream.write(self.prefix + msg)
       return None

    def solve(self, rhs, **kwargs):
        """
        This is the :meth:`solve` method of the abstract KrylovMethod class.
        The class must be specialized and this method overridden.
        """
        raise NotImplementedError, 'This method must be subclassed'
