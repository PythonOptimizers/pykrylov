.. Description of SYMMLQ module
.. _symmlq-page:

=============================================================
The Symmetric Indefinite Method with Orthogonal Factorization
=============================================================

The :mod:`symmlq` Module
========================

.. automodule:: symmlq

.. autoclass:: Symmlq
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


Example
=======

Here is an example using SYMMLQ on a linear system. The coefficient matrix is
read from file in Matrix Market format::

    import numpy as np
    from pykrylov.symmlq import SYMMLQ as KSolver
    from pykrylov.linop import PysparseLinearOperator
    from pysparse import spmatrix
    from pysparse.pysparseMatrix import PysparseMatrix as sp

    A = sp(matrix=spmatrix.ll_mat_from_mtx('lund_a.mtx'))
    n = A.shape[0]
    e = np.ones(n)
    rhs = A*e

    ks = KSolver(PysparseLinearOperator(A))
    ks.solve(rhs, matvec_max=2*n)

    print 'Number of matvecs: ', ks.nMatvec
    print 'Initial/final res: %8.2e/%8.2e' % (ks.residNorm0, ks.residNorm)
    print 'Direct error: %8.2e' % (np.linalg.norm(ks.bestSolution-e)/sqrt(n))


Running this script produces the following output::

    Number of matvecs:  308
    Initial/final res: 1.98e+09/2.83e+01
    Direct error: 2.02e-04
