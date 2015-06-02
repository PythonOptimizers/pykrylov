# -*- coding: utf-8 -*-
"""Cholesky Factorization Operator."""


try:
    from scikits.sparse.cholmod import cholesky
except:
    raise ImportError("Please install scikits.sparse")

from pykrylov.linop import LinearOperator

__docformat__ = 'restructuredtext'


class CholeskyOperator(LinearOperator):
    """Inverse as a Cholesky factorization.

    Linear operator to represent the inverse of a
    symmetric and positive definite sparse matrix
    using its sparse Cholesky factorization.
    """

    def __init__(self, A, **kwargs):
        """CholeskyOperator corresponding to the sparse matrix `A`.

        CholeskyOperator(A)

        The sparse matrix `A` must be in one of the SciPy sparse
        matrix formats.
        """
        m, n = A.shape
        if m != n:
            raise ValueError("Input matrix must be square")

        self.__factor = cholesky(A)

        super(CholeskyOperator, self).__init__(n, n,
                                               matvec=self.cholesky_matvec,
                                               symmetric=True, **kwargs)

    def cholesky_matvec(self, rhs):
        """Solve a linear system with right-hand side `rhs`."""
        return self.__factor(rhs)
