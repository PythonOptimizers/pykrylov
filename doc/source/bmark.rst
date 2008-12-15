.. Some notes on benchmarking solvers on a test set
.. _bmark-page:

====================
Benchmarking Solvers
====================

You may have noticed that in the examples of
sections :ref:`cg-page`, :ref:`cgs-page` and :ref:`bicgstab-page`, the only line
that differs in the example scripts has the form ::

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

    if __name__ == '__main__':

        hdr = '%10s  %6s  %8s  %8s' % ('Name', 'Matvec', 'Resid', 'Error')
        fmt = '%10s  %6d  %8.2e  %8.2e'
        print hdr

        A = sp(matrix=spmatrix.ll_mat_from_mtx('jpwh_991.mtx'))

        n = A.shape[0]
        e = np.ones(n)
        rhs = A*e

        # Loop through solvers using default stopping tolerance
        for KSolver in [CGS, TFQMR, BiCGSTAB]:
            ks = KSolver(lambda v: A*v, matvec_max=2*n)
            ks.solve(rhs, guess = 1+np.arange(n, dtype=np.float))

            err = np.linalg.norm(ks.bestSolution-e)/sqrt(n)
            print fmt % (ks.acronym, ks.nMatvec, ks.residNorm, err)


Executing the script above produces the formatted output::

          Name  Matvec     Resid     Error
           CGS      64  4.72e-03  1.47e-04
         TFQMR      70  6.23e-04  2.77e-05
     Bi-CGSTAB      62  6.34e-03  4.05e-04


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

    if __name__ == '__main__':

        hdr = '%10s  %6s  %8s  %8s' % ('Name', 'Matvec', 'Resid', 'Error')
        fmt = '%10s  %6d  %8.2e  %8.2e'
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
                         matvec_max=2*n
                         precon = lambda u: u/diagA)
            ks.solve(rhs, guess = 1+np.arange(n, dtype=np.float))

            err = np.linalg.norm(ks.bestSolution-e)/sqrt(n)
            print fmt % (ks.acronym, ks.nMatvec, ks.residNorm, err)

.. Why isn't the script above getting colorized???


This time, the output is a bit better than before::

          Name  Matvec     Resid     Error
           CGS      56  3.77e-03  7.42e-05
         TFQMR      59  1.07e-03  6.03e-05
     Bi-CGSTAB      54  4.62e-03  3.02e-04


Much in the same way, a modification of the script above could be used to loop
through preconditioners with a given solver.

.. todo:: Should preconditioners be objects instead of just functions?
