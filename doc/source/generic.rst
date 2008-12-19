.. Description of generic Krylov template
.. _generic-page:

===================================
Generic Template for Krylov Methods
===================================

The :mod:`generic` Module
=========================

.. automodule:: generic

.. autoclass:: KrylovMethod
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


Writing a New Solver
====================

Adding a new solver to `PyKrylov` should be done by
subclassing :class:`KrylovMethod` from the :mod:`generic` module and overriding
the :meth:`solve` method. A general template might look like the following::

    import numpy as np
    from pykrylov.generic import KrylovMethod

    class NewSolver( KrylovMethod ):
        """
        Document your new class and give adequate references.
        """

        def __init__(self, matvec, **kwargs):
            KrylovMethod.__init__(self, matvec, **kwargs)

            self.name = 'New Krylov Solver'
            self.acronym = 'NKS'
            self.prefix = self.acronym + ': '  # Used when verbose=True


        def solve(self, rhs, **kwargs):
            """
            Solve a linear system with `rhs` as right-hand side by the NKS
            method. The vector `rhs` should be a Numpy array. An optional
            argument `guess` may be supplied, with an initial guess as a Numpy
            array. By default, the initial guess is the vector of zeros.
            """
            n = rhs.shape[0]
            nMatvec = 0

            # Initial guess is zero unless one is supplied
            guess_supplied = 'guess' in kwargs.keys()
            x = kwargs.get('guess', np.zeros(n))

            r0 = rhs  # Fixed vector throughout
            if guess_supplied:
                r0 = rhs - self.matvec(x)

            # Further initializations ...

            # Compute initial residual norm. For example:
            residNorm = np.linalg.norm(r0)
            self.residNorm0 = residNorm

            # Compute stopping threshold
            threshold = max( self.abstol, self.reltol * self.residNorm0 )

            if self.verbose:
                self._write('Initial residual = %8.2e\n' % self.residNorm0)
                self._write('Threshold = %8.2e\n' % threshold)

            # Main loop
            while residNorm > threshold and nMatvec < self.matvec_max:

                # Do something ...
                pass

            # End of main loop

            self.nMatvec = nMatvec
            self.bestSolution = x
            self.residNorm = residNorm
