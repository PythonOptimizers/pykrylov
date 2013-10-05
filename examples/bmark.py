# Example benchmarking script
# Requires Pysparse (http://pysparse.sf.net)
# The test matrix may be obtained from http://math.nist.gov/MatrixMarket

import numpy as np
from pykrylov.linop import PysparseLinearOperator
from pykrylov.cgs import CGS
from pykrylov.tfqmr import TFQMR
from pykrylov.bicgstab import BiCGSTAB
from pysparse import spmatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
import sys

class DiagonalPrec:

    def __init__(self, A, **kwargs):
        self.name = 'Diag'
        self.shape = A.shape
        self.diag = np.maximum( np.abs(A.takeDiagonal()), 1.0)

    def __call__(self, y, **kwargs):
        "Return the result of applying preconditioner to y"
        return y/self.diag


if __name__ == '__main__':

    hdr_fmt = '%10s  %6s  %8s  %8s  %8s'
    hdr = hdr_fmt % ('Name', 'Matvec', 'Resid0', 'Resid', 'Error')
    fmt = '%10s  %6d  %8.2e  %8.2e  %8.2e'
    print hdr
    print '-' * len(hdr)

    AA = spmatrix.ll_mat_from_mtx(sys.argv[1])
    A = sp(matrix=AA)
    op = PysparseLinearOperator(A)

    # Create diagonal preconditioner
    dp = DiagonalPrec(A)

    n = A.shape[0]
    e = np.ones(n)
    rhs = A*e

    for KSolver in [CGS, TFQMR, BiCGSTAB]:
        ks = KSolver( op,
                      #precon = dp,
                      #verbose=False,
                      reltol = 1.0e-8
                      )
        ks.solve(rhs, guess = 1+np.arange(n, dtype=op.dtype), matvec_max=2*n)

        err = np.linalg.norm(ks.bestSolution-e)/np.sqrt(n)
        print fmt % (ks.acronym, ks.nMatvec, ks.residNorm0, ks.residNorm, err)

