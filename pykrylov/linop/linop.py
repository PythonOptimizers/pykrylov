from pykrylov.tools.types import *
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

    def __init__(self, nargin, nargout, symmetric=False, hermitian=False, **kwargs):
        self.__nargin = nargin
        self.__nargout = nargout
        self.__symmetric = symmetric
        self.__hermitian = hermitian
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
        "Indicates whether the operator is symmetric."
        return self.__symmetric

    @property
    def hermitian(self):
        "Indicates whether the operator is Hermitian."
        return self.__hermitian

    @property
    def shape(self):
        "The shape of the operator."
        return self.__shape

    @property
    def dtype(self):
        "The data type of the operator."
        return self.__dtype

    @dtype.setter
    def dtype(self, value):
        if value in allowed_types:
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
        if self.hermitian:
            s += ' Hermitian'
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

    def __init__(self, nargin, nargout, matvec, matvec_transp=None, matvec_adj=None, **kwargs):

        super(LinearOperator, self).__init__(nargin, nargout, **kwargs)

        transpose_of = kwargs.pop('transpose_of') if 'transpose_of' in kwargs else None
        adjoint_of = kwargs.pop('adjoint_of') if 'adjoint_of' in kwargs else None
        conjugate_of = kwargs.pop('conjugate_of') if 'conjugate_of' in kwargs else None

        self.__matvec = matvec
        self.__set_transpose(matvec, transpose_of, matvec_transp, **kwargs)
        self.__set_adjoint(matvec, adjoint_of, matvec_adj, **kwargs)

        # For non-complex operators, transpose = adjoint.
        if (self.dtype in integer_types + real_types):
            if self.__T is not None and self.__H is None:
                self.__H = self.__T
            elif self.__T is None and self.__H is not None:
                self.__T = self.__H
        else:
            if transpose_of is None and adjoint_of is None and conjugate_of is None:
                # We're not in a recursive instantiation.
                # Try to infer missing operators.
                __conj = self.conjugate()
                if self.T is not None:
                    self.__T.__H = __conj
                    if self.H is None and __conj is not None:
                        self.logger.debug('Inferring .H')
                        self.__H = __conj.T
                if self.H is not None:
                    self.__H.__T = __conj
                    if self.T is None and __conj is not None:
                        self.logger.debug('Inferring .T')
                        self.__T = __conj.H

    def __set_transpose(self, matvec, transpose_of=None, matvec_transp=None, **kwargs):

        self.__T = None
        if self.symmetric:
            self.__T = self
            return

        if transpose_of is None:
            if matvec_transp is not None:
                # Create 'pointer' to transpose operator.
                self.__T = LinearOperator(self.nargout, self.nargin,
                                          matvec_transp,
                                          matvec_transp=matvec,
                                          transpose_of=self,
                                          **kwargs)
        else:
            # Use operator supplied as transpose operator.
            if isinstance(transpose_of, BaseLinearOperator):
                self.__T = transpose_of
            else:
                msg = 'kwarg transpose_of must be a BaseLinearOperator.'
                msg += ' Got ' + str(transpose_of.__class__)
                raise ValueError(msg)

    def __set_adjoint(self, matvec, adjoint_of=None, matvec_adj=None, **kwargs):

        self.__H = None
        if self.hermitian:
            self.__H = self
            return

        if adjoint_of is None:
            if matvec_adj is not None:
                # Create 'pointer' to adjoint operator.
                self.__H = LinearOperator(self.nargout, self.nargin,
                                          matvec_adj,
                                          matvec_adj=matvec,
                                          adjoint_of=self,
                                          **kwargs)
        else:
            # Use operator supplied as adjoint operator.
            if isinstance(adjoint_of, BaseLinearOperator):
                self.__H = adjoint_of
            else:
                msg = 'kwarg adjoint_of must be a BaseLinearOperator.'
                msg += ' Got ' + str(adjoint_of.__class__)
                raise ValueError(msg)

    @property
    def T(self):
        "The transpose operator."
        return self.__T

    @property
    def H(self):
        "The adjoint operator."
        return self.__H

    @property
    def bar(self):
        "The complex conjugate operator."
        return self.conjugate()

    def conjugate(self):
        "Return the complex conjugate operator."
        if not self.dtype in complex_types:
            return self

        # conj(A) * x = conj(A * conj(x))
        def matvec(x):
            if x.dtype not in complex_types:
                return (self * x).conjugate()
            return (self * x.conjugate()).conjugate()

        # conj(A).T * x = A.H * x = conj(A.T * conj(x))
        # conj(A).H * x = A.T * x = conj(A.H * conj(x))
        if self.H is not None:
            matvec_transp = self.H.__matvec

            if self.T is not None:
                matvec_adj = self.T.__matvec
            else:
                def matvec_adj(x):
                    if x.dtype not in complex_types:
                        return (self.H * x).conjugate()
                    return (self.H * x.conjugate()).conjugate()

        elif self.T is not None:
            matvec_adj = self.T.__matvec

            def matvec_transp(x):
                if x.dtype not in complex_types:
                    return (self.T * x).conjugate()
                return (self.T * x.conjugate()).conjugate()

        else:
            # Cannot infer transpose or adjoint of conjugate operator.
            matvec_transp = matvec_adj = None

        return LinearOperator(self.nargin, self.nargout,
                              matvec=matvec,
                              matvec_transp=matvec_transp,
                              matvec_adj=matvec_adj,
                              transpose_of=self.H,
                              adjoint_of=self.T,
                              conjugate_of=self,
                              dtype=self.dtype)

    def to_array(self):
        "Convert operator to a dense matrix. This is the same as `full`."
        n, m = self.shape
        H = np.empty((n, m), dtype=self.dtype)
        e = np.zeros(m, dtype=self.dtype)
        for j in xrange(m):
            e[j] = 1
            H[:, j] = self * e
            e[j] = 0
        return H

    def full(self):
        "Convert operator to a dense matrix. This is the same as `to_array`."
        return self.to_array()

    def _matvec(self, x):
        """
        Matrix-vector multiplication.

        Encapsulates the matvec routine specified at
        construct time, to ensure the consistency of the input and output
        arrays with the operator's shape.
        """
        x = np.asanyarray(x)
        nargout, nargin = self.shape

        # check input data consistency
        try:
            x = x.reshape(nargin)
        except ValueError:
            msg = 'input array size incompatible with operator dimensions'
            raise ValueError(msg)

        y = self.__matvec(x)

        # check output data consistency
        try:
            y = y.reshape(nargout)
        except ValueError:
            msg = 'output array size incompatible with operator dimensions'
            raise ValueError(msg)

        return y

    def rmatvec(self, x):
        """
        Product with the conjugate transpose. This method is included for
        compatibility with Scipy only. Please use the `H` attribute instead.
        """
        return self.__H.__mul__(x)

    def __mul_scalar(self, x):
        "Product between a linear operator and a scalar."
        result_type = np.result_type(self.dtype, type(x))

        if x == 0:
            return ZeroOperator(self.nargin, self.nargout,
                                dtype=result_type)

        def matvec(y):
            return x * (self(y))

        def matvec_transp(y):
            return x * (self.T(y))

        def matvec_adj(y):
            return x.conjugate() * (self.H(y))

        return LinearOperator(self.nargin, self.nargout,
                              symmetric=self.symmetric,
                              hermitian=(result_type not in complex_types) and self.hermitian,
                              matvec=matvec,
                              matvec_transp=matvec_transp,
                              matvec_adj=matvec_adj,
                              dtype=result_type)

    def __mul_linop(self, op):
        "Product between two linear operators."
        if self.nargin != op.nargout:
            raise ShapeError('Cannot multiply operators together')

        def matvec(x):
            return self(op(x))

        def matvec_transp(x):
            return op.T(self.T(x))

        def matvec_adj(x):
            return op.H(self.H(x))

        result_type = np.result_type(self.dtype, op.dtype)

        return LinearOperator(op.nargin, self.nargout,
                              symmetric=False,   # Generally.
                              hermitian=False,   # Generally.
                              matvec=matvec,
                              matvec_transp=matvec_transp,
                              matvec_adj=matvec_adj,
                              dtype=result_type)

    def __mul_vector(self, x):
        "Product between a linear operator and a vector."
        self._nMatvec += 1
        result_type = np.result_type(self.dtype, x.dtype)
        return self._matvec(x).astype(result_type)

    def __mul__(self, x):
        if np.isscalar(x):
            return self.__mul_scalar(x)
        if isinstance(x, BaseLinearOperator):
            return self.__mul_linop(x)
        if isinstance(x, np.ndarray):
            return self.__mul_vector(x)
        raise ValueError('Cannot multiply')

    def __rmul__(self, x):
        if np.isscalar(x):
            return self.__mul_scalar(x)
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

        def matvec_adj(x):
            return self.H(x) + other.H(x)

        result_type = np.result_type(self.dtype, other.dtype)

        return LinearOperator(self.nargin, self.nargout,
                              symmetric=self.symmetric and other.symmetric,
                              hermitian=self.hermitian and other.hermitian,
                              matvec=matvec,
                              matvec_transp=matvec_transp,
                              matvec_adj=matvec_adj,
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

        def matvec_adj(x):
            return self.H(x) - other.H(x)

        result_type = np.result_type(self.dtype, other.dtype)

        return LinearOperator(self.nargin, self.nargout,
                              symmetric=self.symmetric and other.symmetric,
                              hermitian=self.hermitian and other.hermitian,
                              matvec=matvec,
                              matvec_transp=matvec_transp,
                              matvec_adj=matvec_adj,
                              dtype=result_type)

    def __div__(self, other):
        if not np.isscalar(other):
            raise ValueError('Cannot divide')
        return self * (1 / other)

    def __truediv__(self, other):
        if not np.isscalar(other):
            raise ValueError('Cannot divide')
        return self * (1. / other)

    def __pow__(self, other):
        if not isinstance(other, int):
            raise ValueError('Can only raise to integer power')
        if other < 0:
            raise ValueError('Can only raise to nonnegative power')
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
                                               hermitian=True,
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
        if 'hermitian' in kwargs:
            kwargs.pop('hermitian')
        if 'matvec' in kwargs:
            kwargs.pop('matvec')
        if 'matvec_adj' in kwargs:
            kwargs.pop('matvec_adj')
        if 'dtype' in kwargs:
            kwargs.pop('dtype')

        diag = np.asarray(diag)
        if diag.ndim != 1:
            raise ValueError('Input must be 1-d array')
        self.__diag = diag

        super(DiagonalOperator, self).__init__(diag.shape[0], diag.shape[0],
                                               symmetric=True,
                                               hermitian=(diag.dtype not in complex_types),
                                               matvec=lambda x: diag*x,
                                               matvec_adj=lambda x: diag.conjugate()*x,
                                               dtype=diag.dtype,
                                               **kwargs)

    def __abs__(self):
        return DiagonalOperator(np.abs(self.__diag))

    def _sqrt(self):
        if self.dtype not in complex_types and np.any(self.__diag < 0):
            raise ValueError('Math domain error')
        return DiagonalOperator(np.sqrt(self.__diag))


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
            result_type = np.result_type(self.dtype, x.dtype)
            return np.zeros(nargout, dtype=result_type)

        def matvec_transp(x):
            if x.shape != (nargout,):
                msg = 'Input has shape ' + str(x.shape)
                msg += ' instead of (%d,)' % self.nargout
                raise ValueError(msg)
            result_type = np.result_type(self.dtype, x.dtype)
            return np.zeros(nargin, dtype=result_type)

        super(ZeroOperator, self).__init__(nargin, nargout,
                                           symmetric=(nargin == nargout),
                                           matvec=matvec,
                                           matvec_transp=matvec_transp,
                                           matvec_adj=matvec_transp,
                                           **kwargs)

    def __abs__(self):
        return self

    def _sqrt(self):
        return self


def ReducedLinearOperator(op, row_indices, col_indices):
    """
    Reduce a linear operator by limiting its input to `col_indices` and its
    output to `row_indices`.
    """

    nargin, nargout = len(col_indices), len(row_indices)
    m, n = op.shape    # Shape of non-reduced operator.

    def matvec(x):
        z = np.zeros(n, dtype=x.dtype) ; z[col_indices] = x[:]
        y = op * z
        return y[row_indices]

    def matvec_transp(x):
        z = np.zeros(m, dtype=x.dtype) ; z[row_indices] = x[:]
        y = op.T * z
        return y[col_indices]

    def matvec_adj(x):
        z = np.zeros(m, dtype=x.dtype) ; z[row_indices] = x[:]
        y = op.H * z
        return y[col_indices]

    return LinearOperator(nargin, nargout,
                          symmetric=False,
                          hermitian=False,
                          matvec=matvec,
                          matvec_transp=matvec_transp,
                          matvec_adj=matvec_adj,
                          dtype=op.dtype)


def SymmetricallyReducedLinearOperator(op, indices):
    """
    Reduce a linear operator symmetrically by reducing boths its input and
    output to `indices`.
    """

    nargin = len(indices)
    m, n = op.shape    # Shape of non-reduced operator.

    def matvec(x):
        z = np.zeros(n, dtype=x.dtype) ; z[indices] = x[:]
        y = op * z
        return y[indices]

    def matvec_transp(x):
        z = np.zeros(m, dtype=x.dtype) ; z[indices] = x[:]
        y = op.T * z
        return y[indices]

    def matvec_adj(x):
        z = np.zeros(m, dtype=x.dtype) ; z[indices] = x[:]
        y = op.H * z
        return y[indices]

    return LinearOperator(nargin, nargin,
                          symmetric=op.symmetric,
                          hermitian=op.hermitian,
                          matvec=matvec,
                          matvec_transp=matvec_transp,
                          matvec_adj=matvec_adj,
                          dtype=op.dtype)


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


def linop_from_ndarray(A, symmetric=False, **kwargs):
    "Return a linear operator from a Numpy `ndarray`."

    hermitian = kwargs.get('hermitian', symmetric)

    if A.dtype in complex_types:
        return LinearOperator(A.shape[1], A.shape[0],
                              lambda v: np.dot(A, v),
                              matvec_transp=lambda u: np.dot(A.T, u),
                              matvec_adj=lambda w: np.dot(A.conjugate().T, w),
                              symmetric=symmetric,
                              hermitian=hermitian,
                              dtype=A.dtype)

    if symmetric ^ hermitian:
        raise ValueError('For non-complex operators, transpose = adjoint.')

    return LinearOperator(A.shape[1], A.shape[0],
                          lambda v: np.dot(A, v),
                          matvec_transp=lambda u: np.dot(A.T, u),
                          symmetric=symmetric or hermitian,
                          hermitian=symmetric or hermitian,
                          dtype=A.dtype)


def sqrt(op):
    "Return the square root of a linear operator, if defined."
    return op._sqrt()

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
