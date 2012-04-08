# Common code for demos.

import numpy as np
from pysparse import spmatrix
from pykrylov.linop import PysparseLinearOperator as Op
from math import sqrt
import sys

def demo(KSolver, mtx, **kwargs):

    hdr_fmt = '%10s  %6s  %8s  %8s  %8s'
    hdr = hdr_fmt % ('Name', 'Matvec', 'Resid0', 'Resid', 'Error')
    fmt = '%10s  %6d  %8.2e  %8.2e  %8.2e'

    AA = spmatrix.ll_mat_from_mtx(mtx)
    A = Op(AA)

    n = A.get_shape()[0]
    e = np.ones(n)
    rhs = A*e

    if 'logger' in kwargs:
        logger = kwargs.pop('logger')

    ks = KSolver(A, reltol=1.0e-8, logger=logger)
    ks.solve(rhs, guess=1+np.arange(n, dtype=np.float), matvec_max=2*n,
             **kwargs)

    err = np.linalg.norm(ks.bestSolution-e)/sqrt(n)

    print
    print hdr
    print '-' * len(hdr)
    print fmt % (ks.acronym, ks.nMatvec, ks.residNorm0, ks.residNorm, err)
