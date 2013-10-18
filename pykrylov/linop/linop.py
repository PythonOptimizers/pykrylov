import numpy as np
import logging

__docformat__ = 'restructuredtext'


# Default (null) logger.
null_log = logging.getLogger('linop')
null_log.setLevel(logging.INFO)
null_log.addHandler(logging.NullHandler())


class BaseLinearOperator(object):
    """
    A linear operator is a linear mapping x -> A(x) such that the size of the
    input vector x is `nargin` and the size of the output is `nargout`. It can
    be visualized as a matrix of shape (`nargout`, `nargin`). Its type is any
    valid Numpy `dtype`. By default, it has `dtype` `numpy.float` but this can
    be changed to, e.g., `numpy.complex` via the `dtype` keyword argument and
    attribute.

    A logger may be attached to the linear operator via the `logger` keyword
    argument.
    """

    def __init__(self, nargin, nargout, symmetric=False, **kwargs):
        self.__nargin = nargin
        self.__nargout = nargout
        self.__symmetric = symmetric
        self.__shape = (nargout, nargin)
        self.__dtype = kwargs.get('dtype', np.float)
        self._nMatvec = 0

        # Log activity.
        self.logger = kwargs.get('logger', null_log)
        self.logger.info('New linear operator with shape ' + str(self.shape))
        return

    @property
    def nargin(self):
        "The size of an input vector."
        return self.__nargin

    @property
    def nargout(self):
        "The size of an output vector."
        return self.__nargout

    @property
    def symmetric(self):
        "Indicates whether the operator is symmetric or not."
        return self.__symmetric

    @property
    def shape(self):
        "The shape of the operator."
        return self.__shape

    @property
    def dtype(self):
        return self.__dtype

    @dtype.setter
    def dtype(self, value):
        allowed_types = np.core.numerictypes.typeDict.keys() + \
                            [np.float, np.complex, np.int, np.uint]
        if value in allows_types:
            self.__dtype = value
        else:
            raise TypeError('Not a Numpy type')

    @property
    def nMatvec(self):
        "The number of products with vectors computed so far."
        return self._nMatvec

    def reset_counters(self):
        "Reset operator/vector product counter to zero."
        self._nMatvec = 0

    def __call__(self, *args, **kwargs):
        # An alias for __mul__.
        return self.__mul__(*args, **kwargs)

    def __mul__(self, x):
        raise NotImplementedError('Please subclass to implement __mul__.')

    def __repr__(self):
        if self.symmetric:
            s = 'Symmetric'
        else:
            s = 'Unsymmetric'
        s += ' <' + self.__class__.__name__ + '>'
        s += ' of type %s' % self.dtype
        s += ' with shape (%d,%d)' % (self.nargout, self.nargin)
        return s


class LinearOperator(BaseLinearOperator):
    """
    A linear operator constructed from a `matvec` and (possibly) a
    `matvec_transp` function. If `symmetric` is `True`, `matvec_transp` is
    ignored. All other keyword arguments are passed directly to the superclass.
    """

    def __init__(self, nargin, nargout, matvec, matvec_transp=None, **kwargs):

        super(LinearOperator, self).__init__(nargin, nargout, **kwargs)
        self.__transposed = kwargs.get('transposed', False)
        transpose_of = kwargs.get('transpose_of', None)

        self.__matvec = matvec

        if self.symmetric:
            self.__T = self
        else:
            if transpose_of is None:
                if matvec_transp is not None:
                    # Create 'pointer' to transpose operator.
                    self.__T = LinearOperator(nargout, nargin,
                                              matvec_transp,
                                              matvec_transp=matvec,
                                              transposed=not self.__transposed,
                                              transpose_of=self,
                                              **kwargs)
                else:
                    self.__T = None
            else:
                # Use operator supplied as transpose operator.
                if isinstance(transpose_of, BaseLinearOperator):
                    self.__T = transpose_of
                else:
                    msg = 'kwarg transposed_of must be a BaseLinearOperator.'
                    msg += ' Got ' + str(transpose_of.__class__)
                    raise ValueError(msg)

    @property
    def T(self):
        "The transpose operator."
        return self.__T

    def to_array(self):
        n,m = self.shape
        H = np.empty((n,m))
        for j in xrange(m):
            ej = np.zeros(m) ; ej[j] = 1.0
            H[:,j] = self * ej
        return H

    def __mul_scalar(self, x):
        "Product between a linear operator and a scalar."
        def matvec(y):
            return x * (self(y))

        def matvec_transp(y):
            return x * (self.T(y))

        result_type = np.result_type(self.dtype, type(x))

        return LinearOperator(self.nargin, self.nargout,
                              symmetric=self.symmetric,
                              matvec=matvec,
                              matvec_transp=matvec_transp,
                              dtype=result_type)

    def __mul_linop(self, op):
        "Product between two linear operators."
        if self.nargin != op.nargout:
            raise ShapeError('Cannot multiply operators together')

        def matvec(x):
            return self(op(x))

        def matvec_transp(x):
            return op.T(self.T(x))

        result_type = np.result_type(self.dtype, op.dtype)

        return LinearOperator(op.nargin, self.nargout,
                              symmetric=False,   # Generally.
                              matvec=matvec,
                              matvec_transp=matvec_transp,
                              dtype=result_type)

    def __mul_vector(self, x):
        "Product between a linear operator and a vector."
        self._nMatvec += 1
        result_type = np.result_type(self.dtype, x.dtype)
        return self.__matvec(x).astype(result_type)

    def __mul__(self, x):
        if np.isscalar(x):
            return self.__mul_scalar(x)
        if isinstance(x, BaseLinearOperator):
            return self.__mul_linop(x)
        if isinstance(x, np.ndarray):
            return self.__mul_vector(x)
        raise ValueError('Cannot multiply')

    def __rmul__(self, x):
        if isinstance(x, BaseLinearOperator):
            return x.__mul_linop(self)
        raise ValueError('Cannot multiply')

    def __add__(self, other):
        if not isinstance(other, BaseLinearOperator):
            raise ValueError('Cannot add')
        if self.shape != other.shape:
            raise ShapeError('Cannot add')

        def matvec(x):
            return self(x) + other(x)

        def matvec_transp(x):
            return self.T(x) + other.T(x)

        result_type = np.result_type(self.dtype, other.dtype)

        return LinearOperator(self.nargin, self.nargout,
                                  symmetric=self.symmetric and other.symmetric,
                                  matvec=matvec,
                                  matvec_transp=matvec_transp,
                                  dtype=result_type)

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        if not isinstance(other, BaseLinearOperator):
            raise ValueError('Cannot add')
        if self.shape != other.shape:
            raise ShapeError('Cannot add')

        def matvec(x):
            return self(x) - other(x)

        def matvec_transp(x):
            return self.T(x) - other.T(x)

        result_type = np.result_type(self.dtype, other.dtype)

        return LinearOperator(self.nargin, self.nargout,
                                  symmetric=self.symmetric and other.symmetric,
                                  matvec=matvec,
                                  matvec_transp=matvec_transp,
                                  dtype=result_type)

    def __div__(self, other):
        if not np.isscalar(other):
            raise ValueError('Cannot divide')
        return self * (1./other)

    def __pow__(self, other):
        if not isinstance(other, int):
            raise ShapeError('Can only raise to integer power')
        if other < 0:
            raise ShapeError('Can only raise to nonnegative power')
        if self.nargin != self.nargout:
            raise ShapeError('Can only raise square operators to a power')
        if other == 0:
            return IdentityOperator(self.nargin)
        if other == 1:
            return self
        return self * self**(other-1)


class IdentityOperator(LinearOperator):
    """
    A linear operator representing the identity operator of size `nargin`.
    """

    def __init__(self, nargin, **kwargs):
        if 'symmetric' in kwargs:
            kwargs.pop('symmetric')
        if 'matvec' in kwargs:
            kwargs.pop('matvec')

        super(IdentityOperator, self).__init__(nargin, nargin,
                                               symmetric=True,
                                               matvec=lambda x: x,
                                               **kwargs)


class DiagonalOperator(LinearOperator):
    """
    A diagonal linear operator defined by its diagonal `diag` (a Numpy array.)
    The type must be specified in the `diag` argument, e.g.,
    `np.ones(5, dtype=np.complex)` or `np.ones(5).astype(np.complex)`.
    """

    def __init__(self, diag, **kwargs):
        if 'symmetric' in kwargs:
            kwargs.pop('symmetric')
        if 'matvec' in kwargs:
            kwargs.pop('matvec')
        if 'dtype' in kwargs:
            kwargs.pop('dtype')

        super(DiagonalOperator, self).__init__(diag.shape[0], diag.shape[0],
                                               symmetric=True,
                                               matvec=lambda x: diag*x,
                                               dtype=diag.dtype,
                                               **kwargs)


class ZeroOperator(LinearOperator):
    """
    The zero linear operator of shape `nargout`-by-`nargin`.
    """

    def __init__(self, nargin, nargout, **kwargs):
        if 'matvec' in kwargs:
            kwargs.pop('matvec')
        if 'matvec_transp' in kwargs:
            kwargs.pop('matvec_transp')

        def matvec(x):
            if x.shape != (nargin,):
                msg = 'Input has shape ' + str(x.shape)
                msg += ' instead of (%d,)' % self.nargin
                raise ValueError(msg)
            return np.zeros(nargout)

        def matvec_transp(x):
            if x.shape != (nargout,):
                msg = 'Input has shape ' + str(x.shape)
                msg += ' instead of (%d,)' % self.nargout
                raise ValueError(msg)
            return np.zeros(nargin)

        super(ZeroOperator, self).__init__(nargin, nargout,
                                           matvec=matvec,
                                           matvec_transp=matvec_transp,
                                           **kwargs)


def ReducedLinearOperator(op, row_indices, col_indices):
    """
    Reduce a linear operator by limiting its input to `col_indices` and its
    output to `row_indices`.
    """

    nargin, nargout = len(col_indices), len(row_indices)
    m, n = op.shape    # Shape of non-reduced operator.

    def matvec(x):
        z = np.zeros(n) ; z[col_indices] = x[:]
        y = op * z
        return y[row_indices]

    def matvec_transp(x):
        z = np.zeros(m) ; z[row_indices] = x[:]
        y = op.T * z
        return y[col_indices]

    return LinearOperator(nargin, nargout, matvec=matvec, symmetric=False,
                          matvec_transp=matvec_transp)


def SymmetricallyReducedLinearOperator(op, indices):
    """
    Reduce a linear operator symmetrically by reducing boths its input and
    output to `indices`.
    """

    nargin = len(indices)
    m, n = op.shape    # Shape of non-reduced operator.

    def matvec(x):
        z = np.zeros(n) ; z[indices] = x[:]
        y = op * z
        return y[indices]

    def matvec_transp(x):
        z = np.zeros(m) ; z[indices] = x[:]
        y = op * z
        return y[indices]

    return LinearOperator(nargin, nargin, matvec=matvec,
                          symmetric=op.symmetric, matvec_transp=matvec_transp)


class ShapeError(Exception):
    """
    Exception raised when defining a linear operator of the wrong shape or
    multiplying a linear operator with a vector of the wrong shape.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def PysparseLinearOperator(A):
    "Return a linear operator from a Pysparse sparse matrix."

    nargout, nargin = A.shape
    try:
        symmetric = A.issym
    except:
        symmetric = A.isSymmetric()

    def matvec(x):
        if x.shape != (nargin,):
            msg = 'Input has shape ' + str(x.shape)
            msg += ' instead of (%d,)' % nargin
            raise ValueError(msg)
        if hasattr(A, '__mul__'):
            return A*x
        Ax = np.empty(nargout)
        A.matvec(x, Ax)
        return Ax

    def matvec_transp(y):
        if y.shape != (nargout,):
            msg = 'Input has shape ' + str(y.shape)
            msg += ' instead of (%d,)' % nargout
            raise ValueError(msg)
        if hasattr(A, '__rmul__'):
            return y*A
        ATy = np.empty(nargin)
        A.matvec_transp(y, ATy)
        return ATy

    return LinearOperator(nargin, nargout, matvec=matvec,
                          matvec_transp=matvec_transp, symmetric=symmetric)


def linop_from_ndarray(A):
    "Return a linear operator from a Numpy `ndarray`."
    return LinearOperator(A.shape[1], A.shape[0],
                          lambda v: np.dot(A, v),
                          matvec_transp=lambda u: np.dot(A.T, u),
                          symmetric=False)


if __name__ == '__main__':
    from pykrylov.tools import check_symmetric
    from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
    from nlpy.model import AmplModel
    import sys

    np.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

    nlp = AmplModel(sys.argv[1])
    J = sp(matrix=nlp.jac(nlp.x0))
    e1 = np.ones(J.shape[0])
    e2 = np.ones(J.shape[1])
    print 'J.shape = ', J.getShape()

    print 'Testing PysparseLinearOperator:'
    op = PysparseLinearOperator(J)
    print 'op.shape = ', op.shape
    print 'op.T.shape = ', op.T.shape
    print 'op * e2 = ', op * e2
    print "op.T * e1 = ", op.T * e1
    print 'op.T.T * e2 = ', op.T.T * e2
    print 'op.T.T.T * e1 = ', op.T.T.T * e1
    print 'With call:'
    print 'op(e2) = ', op(e2)
    print 'op.T(e1) = ', op.T(e1)
    print 'op.T.T is op : ', (op.T.T is op)
    print
    print 'Testing LinearOperator:'
    op = LinearOperator(J.shape[1], J.shape[0],
                        lambda v: J*v,
                        matvec_transp=lambda u: u*J)
    print 'op.shape = ', op.shape
    print 'op.T.shape = ', op.T.shape
    print 'op * e2 = ', op * e2
    print 'e1.shape = ', e1.shape
    print 'op.T * e1 = ', op.T * e1
    print 'op.T.T * e2 = ', op.T.T * e2
    print 'op(e2) = ', op(e2)
    print 'op.T(e1) = ', op.T(e1)
    print 'op.T.T is op : ', (op.T.T is op)
    print
    op2 = op.T * op
    print 'op2 * e2 = ', op2 * e2
    print 'op.T * (op * e2) = ', op.T * (op * e2)
    print 'op2 is symmetric: ', check_symmetric(op2)
    op3 = op * op.T
    print 'op3 * e1 = ', op3 * e1
    print 'op * (op.T * e1) = ', op * (op.T * e1)
    print 'op3 is symmetric: ', check_symmetric(op3)
    print
    print 'Testing negative operator:'
    nop = -op
    print op * e2
    print nop * e2
