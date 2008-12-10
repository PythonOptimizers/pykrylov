.. Description of conjugate gradient module
.. _cg-page:

=============================
The Conjugate Gradient Method
=============================

The :mod:`cg` Module
====================

.. automodule:: cg

.. autoclass:: CG
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


Example
=======

The following script runs through a list of symmetric positive definite matrices
in `Matrix Market <http://math.nist.gov/MatrixMarket>`_ format, builds
the right-hand side so that the exact solution of the system is the vector of
ones, and applies the conjugate gradient to each of them. The sparse matrices
are handled by way of the `Pysparse <http://pysparse.sf.net>`_ package. ::

    from pykrylov.cg import CG
    from pysparse import spmatrix
    from pysparse.pysparseMatrix import PysparseMatrix as sp
    import numpy as np
    from math import sqrt

    matrices = ['1138bus.mtx', 'bcsstk08.mtx', 'bcsstk09.mtx', 'bcsstk10.mtx']
    matrices += ['bcsstk11.mtx', 'bcsstk18.mtx', 'bcsstk19.mtx' ]

    hdr = '%15s  %5s  %5s  %8s  %8s  %8s' % ('Name', 'Size', 'Mult',
                                             'Resid0', 'Resid', 'Error')
    print hdr

    for matName in matrices:
        A = sp( matrix=spmatrix.ll_mat_from_mtx(matName) )
        n = A.shape[0]
        e = np.ones(n)
        rhs = A * e
        cg = CG(lambda v: A*v, matvec_max=2*n)
        cg.solve(rhs)
        err = np.linalg.norm(cg.bestSolution - e)/sqrt(n)
        print '%15s  %5d  %5d  %8.2e  %8.2e  %8.2e' % (matName, n, cg.nMatvec,
                                                       cg.residNorm0,
                                                       cg.residNorm, err)


The script above produces the following output::

               Name   Size   Mult    Resid0     Resid     Error
        1138bus.mtx   1138   1759  1.46e+03  1.44e-03  1.30e-05
       bcsstk08.mtx   1074   1255  8.74e+10  7.41e+04  7.54e-02
       bcsstk09.mtx   1083    180  3.17e+08  1.84e+02  8.59e-06
       bcsstk10.mtx   1086   1753  8.10e+07  7.30e+01  1.04e-03
       bcsstk11.mtx   1473   1689  5.43e+09  4.88e+03  5.68e-02
       bcsstk18.mtx  11948   9729  2.79e+11  2.49e+05  2.85e-01
       bcsstk19.mtx    817    468  1.34e+15  1.26e+09  8.09e-01

Note that the default relative stopping tolerance is `1.0e-6` and is achieved in
all cases.
