.. Description of transpose-free quasi-minimal residual module
.. _tfqmr-page:

================================================
The Transpose-Free Quasi-Minimal Residual Method
================================================

The :mod:`tfqmr` Module
=======================

.. automodule:: tfqmr

.. autoclass:: TFQMR
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


Example
=======

Here is an example using TFQMR on a linear system. The coefficient matrix is
read from file in Matrix Market format::

    import numpy as np
    from pykrylov.tfqmr import TFQMR as KSolver
    from pykrylov.linop import PysparseLinearOperator
    from pysparse import spmatrix
    from pysparse.pysparseMatrix import PysparseMatrix as sp

    A = sp(matrix=spmatrix.ll_mat_from_mtx('jpwh_991.mtx'))
    n = A.shape[0]
    e = np.ones(n)
    rhs = A*e

    ks = KSolver( PysparseLinearOperator(A),
                  matvec_max=2*n,
                  verbose=False,
                  reltol = 1.0e-5 )
    ks.solve(rhs, guess = 1+np.arange(n, dtype=np.float))

    print 'Number of matvecs: ', ks.nMatvec
    print 'Initial/final res: %8.2e/%8.2e' % (ks.residNorm0, ks.residNorm)
    print 'Direct error: %8.2e' % (np.linalg.norm(ks.bestSolution-e)/sqrt(n))


Running this script produces the following output::

    Number of matvecs:  70
    Initial/final res: 8.64e+03/6.23e-04
    Direct error: 2.77e-05
