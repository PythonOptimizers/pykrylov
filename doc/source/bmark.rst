.. Some notes on benchmarking solvers on a test set
.. _bmark-page:

====================
Benchmarking Solvers
====================

You may have noticed that in the examples of sections :ref:`cg-page`,
:ref:`cgs-page`, :ref:`bicgstab-page` and :ref:`tfqmr-page`, the only line that
differs in the example scripts has the form ::

    from pykrylov.tfqmr import TFQMR as KSolver

The rest of the code fragment is exactly the same across all examples. This
similarity could be used, for instance, to benchmark solvers on a set of test
problems.

Consider the script::

    import numpy as np
    from pykrylov.cgs import CGS
    from pykrylov.tfqmr import TFQMR
    from pykrylov.bicgstab import BiCGSTAB
    from pysparse import spmatrix
    from pysparse.pysparseMatrix import PysparseMatrix as sp

    from math import sqrt

    hdr = '%10s  %6s  %8s  %8s  %8s' % ('Name', 'Matvec', 'Resid0', 'Resid', 'Error')
    fmt = '%10s  %6d  %8.2e  %8.2e  %8.2e'
    print hdr

    A = sp(matrix=spmatrix.ll_mat_from_mtx('jpwh_991.mtx'))

    n = A.shape[0]
    e = np.ones(n)
    rhs = A*e

    # Loop through solvers using tighter stopping tolerance
    for KSolver in [CGS, TFQMR, BiCGSTAB]:
        ks = KSolver(lambda v: A*v, matvec_max=2*n, reltol=1.0e-8)
        ks.solve(rhs, guess = 1+np.arange(n, dtype=np.float))

        err = np.linalg.norm(ks.bestSolution-e)/sqrt(n)
        print fmt % (ks.acronym, ks.nMatvec, ks.residNorm0, ks.residNorm, err)


Executing the script above produces the formatted output::

         Name  Matvec    Resid0     Resid     Error
          CGS      82  8.64e+03  3.25e-05  2.35e-06
        TFQMR      84  8.64e+03  8.97e-06  1.22e-06
    Bi-CGSTAB      84  8.64e+03  5.57e-05  4.04e-06


Example with Preconditioning
============================

A preconditioner can be supplied to any Krylov solver via the `precon` keyword
argument upon instantiation.

For example, we could supply a simple (and naive) diagonal preconditioner by
modifying the benchmarking script as::

    import numpy as np
    from pykrylov.cgs import CGS
    from pykrylov.tfqmr import TFQMR
    from pykrylov.bicgstab import BiCGSTAB
    from pysparse import spmatrix
    from pysparse.pysparseMatrix import PysparseMatrix as sp

    from math import sqrt

    hdr = '%10s  %6s  %8s  %8s  %8s' % ('Name', 'Matvec', 'Resid0', 'Resid', 'Error')
    fmt = '%10s  %6d  %8.2e  %8.2e  %8.2e'
    print hdr

    A = sp(matrix=spmatrix.ll_mat_from_mtx('jpwh_991.mtx'))

    # Extract diagonal of A and make it sufficiently positive
    diagA = np.maximum(np.abs(A.takeDiagonal()), 1.0)

    n = A.shape[0]
    e = np.ones(n)
    rhs = A*e

    # Loop through solvers using default stopping tolerance
    for KSolver in [CGS, TFQMR, BiCGSTAB]:
        ks = KSolver(lambda v: A*v,
                     matvec_max=2*n,
                     precon=lambda u: u/diagA,
                     reltol=1.0e-8)
        ks.solve(rhs, guess = 1+np.arange(n, dtype=np.float))

        err = np.linalg.norm(ks.bestSolution-e)/sqrt(n)
        print fmt % (ks.acronym, ks.nMatvec, ks.residNorm, err)

This time, the output is a bit better than before::

          Name  Matvec    Resid0     Resid     Error
           CGS      70  8.64e+03  7.84e-06  2.33e-07
         TFQMR      70  8.64e+03  7.61e-06  2.47e-07
     Bi-CGSTAB      64  8.64e+03  8.54e-05  4.93e-06


Much in the same way, a modification of the script above could be used to loop
through preconditioners with a given solver.

Note that preconditioners need not be functions but can be more general
objects. The only requirement is that they should be callable. For example, the
same effect as above can be achieved by instead defining the preconditioner as::

    class DiagonalPrec:

        def __init__(self, A, **kwargs):
            self.name = 'Diag'
            self.shape = A.shape
            self.diag = np.maximum( np.abs(A.takeDiagonal()), 1.0)

        def __call__(self, y, **kwargs):
            "Return the result of applying preconditioner to y"
            return y/self.diag

If `dp` is an instance of the `DiagonalPrec` class and `y` is a Numpy array of
appropriate size, one solves preconditioning systems by simply calling
`x=dp(y)`. A call to a Krylov solver might thus look like::

    # Create diagonal preconditioner
    dp = DiagonalPrec(A)

    ks = KSolver(lambda v: A*v, matvec_max=2*n, precon=dp, reltol=1.0e-8)

