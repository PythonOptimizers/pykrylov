.. Introduction to PyKrylov

========================
Introduction to PyKrylov
========================

PyKrylov aims to provide a flexible implementation of the most common Krylov
method for solving systems of linear equations in pure Python. The sole
requirement is Numpy for fast array operations.

PyKyrlov is in the *very early* stages of development. If you give it a whirl,
please let me know what you think at the `LightHouse PyKrylov page
<http://pykrylov.lighthouseapp.com/projects/21441-pykrylov>`_. Feel free to post
bug reports, feature requests and praises.

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

    n = 100
    e = np.ones(n)
    rhs = Poisson1dMatvec(e)
    cg = CG(Poisson1dMatvec, matvec_max=200)
    cg.solve(rhs)

    print 'Number of matrix-vector products: ', cg.nMatvec
    print 'Residual: %8.2e' % cg.residNorm
    print 'Error: %8.2e' % (np.linalg.norm(e - CG.bestSolution)/sqrt(n))

On my machine, the above script produces::

    Number of matrix-vector products: 50
    Residual: 7.39e-14
    Error: 2.06e-15
