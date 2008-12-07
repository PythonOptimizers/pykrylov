.. Introduction to PyKrylov

========================
Introduction to PyKrylov
========================

PyKrylov aims to provide a flexible implementation of the most common Krylov
method for solving systems of linear equations in pure Python. The sole
requirement is Numpy for fast array operations.

PyKyrlov is in the *very early* stages of development. If you give it a whirl,
please let me know what you think at `dominique.orban@gmail.com
<mailto:dominique.orban@gmail.com>`_.

Example
=======

The following code snippet solves a linear system with a 1D Poisson coefficient
matrix. Since this matrix is symmetric and positive definite, the conjugate
gradient algorithm is used::

    import numpy as np
    from math import sqrt
    from pykrylov.cg import CG

    def Poisson1dMatvec(x):
        # Matrix-vector product with a 1D Poisson matrix
        y = 2*x
        y[:-1] -= x[1:]
        y[1:] -= x[:-1]
        return y

    e = np.ones(100)
    rhs = Poisson1dMatvec(e)
    cg = CG(Poisson1dMatvec, matvec_max=200)
    cg.solve(rhs)

    print 'Number of matrix-vector products: ', CG.nMatvec
    print 'Residual: %8.2e' % CG.residNorm
    print 'Error: %8.2e' % np.linalg.norm(e - CG.bestSolution)/sqrt(n)

On my machine, the above script produces::

    Number of matrix-vector products: 50
    Residual: 7.39e-14
    Error: 2.06e-15
