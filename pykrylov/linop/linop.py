import numpy as np

__docformat__ = 'restructuredtext'


class LinearOperator:
    """
    A linear operator is a linear mapping x -> A(x) such that the size of the
    input vector x is `nargin` and the size of the output is `nargout`. It can
    be visualized as a matrix of shape (`nargout`, `nargin`).
    """

    def __init__(self, nargin, nargout, **kwargs):
        self.nargin = nargin
        self.nargout = nargout
        self.shape = (nargout, nargin)
        self.nMatvec = 0
        self.nMatvecTransp = 0

        # Log activity.
        self.logger = kwargs.get('logger', None)
        if self.logger is not None:
            self.logger.info('New linear operator with shape ' + str(self.shape))
        return


    def get_shape(self):
        return self.shape


    def check_symmetric(self, loop=10, explicit=False):
        """
        Make sure this linear operator is indeed symmetric. The check is
        performed without using the transpose operator self.T.
        """
        n = self.nargin
        eps = np.finfo(np.double).eps

        if explicit:
            H = np.empty((n,n))
            for i in xrange(n):
                ei = np.zeros(n) ; ei[i] = 1.0
                H[:,i] = self * ei
            if np.linalg.norm(H-H.T) > (eps + np.linalg.norm(H)) * eps**(1.0/3):
                print 'Faulty H:'
                print H
                return False
        else:
            np.random.seed(1)
            for i in xrange(loop):
                y = 10*np.random.random(n)
                w = self * y      # = A*y
                v = self * w      # = A*(A*y)
                s = np.dot(w,w)   # = y'*A'*A*y
                t = np.dot(y,v)   # = y'*(A*(A*y))
                z = abs(s - t)
                epsa = (s + eps) * eps**(1.0/3)

                if self.logger is not None:
                    self.logger.debug("y'*A'*A*y    = %g" % s)
                    self.logger.debug("y'*(A*(A*y)) = %g" % t)
                    self.logger.debug('z = %g, epsa = %g' % (z,epsa))

                if z > epsa:
                    return False
        return True


    def __call__(self, *args, **kwargs):
        # An alias for __mul__.
        return self.__mul__(*args, **kwargs)


    def __mul__(self, x):
        raise NotImplementedError, 'Please subclass to implement __mul__.'



class SimpleLinearOperator(LinearOperator):
    """
    A linear operator constructed from a matvec and (possibly) a matvec_transp
    function.
    """

    def __init__(self, nargin, nargout, matvec,
                 matvec_transp=None, symmetric=False, **kwargs):
        LinearOperator.__init__(self, nargin, nargout, **kwargs)
        self.symmetric = symmetric
        self.transposed = kwargs.get('transposed', False)
        transpose_of = kwargs.get('transpose_of', None)

        self.matvec = matvec

        if symmetric:
            self.T = self
        else:
            if transpose_of is None:
                if matvec_transp is not None:
                    # Create 'pointer' to transpose operator.
                    self.T = SimpleLinearOperator(nargout, nargin,
                                                  matvec_transp,
                                                  matvec_transp=matvec,
                                                  transposed=not self.transposed,
                                                  transpose_of=self,
                                                  logger=kwargs.get('logger',None))
                else:
                    self.T = None
            else:
                # Use operator supplied as transpose operator.
                if isinstance(transpose_of, LinearOperator):
                    self.T = transpose_of
                else:
                    msg = 'kwarg transposed_of must be a LinearOperator.'
                    msg += ' Got ' + str(transpose_of.__class__)
                    raise ValueError, msg

    def __mul__(self, x):
        if self.transposed:
            self.nMatvecTransp += 1
        else:
            self.nMatvec += 1
        return self.matvec(x)



class PysparseLinearOperator(LinearOperator):
    """
    A linear operator constructed from any object implementing either `__mul__`
    or `matvec` and either `__rmul__` or `matvec_transp`, such as a `ll_mat`
    object or a `PysparseMatrix` object.
    """

    def __init__(self, A, symmetric=False, **kwargs):
        m, n = A.shape
        self.A = A
        self.symmetric = symmetric
        self.transposed = kwargs.get('transposed', False)
        transpose_of = kwargs.get('transpose_of', None)

        if self.transposed:

            LinearOperator.__init__(self, m, n, **kwargs)
            self.__mul__ = self._rmul

        else:

            LinearOperator.__init__(self, n, m, **kwargs)
            self.__mul__ = self._mul

        if self.logger is not None:
            self.logger.info('New linop has transposed='+str(self.transposed))

        if symmetric:
            self.T = self
        else:
            if transpose_of is None:
                # Create 'pointer' to transpose operator.
                self.T = PysparseLinearOperator(self.A,
                                                transposed=not self.transposed,
                                                transpose_of=self,
                                                logger=self.logger)
            else:
                # Use operator supplied as transpose operator.
                if isinstance(transpose_of, LinearOperator):
                    self.T = transpose_of
                else:
                    msg = 'kwarg transposed_of must be a LinearOperator.'
                    msg += ' Got ' + str(transpose_of.__class__)
                    raise ValueError, msg

        return


    def _mul(self, x):
        # Make provision for the case where A does not implement __mul__.
        if x.shape != (self.nargin,):
            msg = 'Input has shape ' + str(x.shape)
            msg += ' instead of (%d,)' % self.nargin
            raise ValueError, msg
        if self.transposed:
            self.nMatvecTransp += 1
        else:
            self.nMatvec += 1
        if hasattr(self.A, '__mul__'):
            return self.A.__mul__(x)
        Ax = np.empty(self.nargout)
        self.A.matvec(x, Ax)
        return Ax


    def _rmul(self, y):
        # Make provision for the case where A does not implement __rmul__.
        # This method is only relevant when transposed=True.
        if y.shape != (self.nargin,):  # This is the transposed op's nargout!
            msg = 'Input has shape ' + str(y.shape)
            msg += ' instead of (%d,)' % self.nargin
            raise ValueError, msg
        if self.transposed:
            self.nMatvec += 1
        else:
            self.nMatvecTransp += 1
        if hasattr(self.A, '__rmul__'):
            return self.A.__rmul__(y)
        ATy = np.empty(self.nargout)   # This is the transposed op's nargin!
        self.A.matvec_transp(y, ATy)
        return ATy


# It would be much better if we could add and multiply linear operators.
# In the meantime, here is a patch.
class SquaredLinearOperator(LinearOperator):
    """
    Given a linear operator ``A``, build the linear operator ``A.T * A``. If
    ``transpose`` is set to ``True``, build ``A * A.T`` instead. This may be
    useful for solving one of the normal equations

    |           A'Ax = A'b
    |           AA'y = Ag

    which are the optimality conditions of the linear least-squares problems

    |          minimize{in x}  | Ax-b |
    |          minimize{in y}  | A'y-g |

    in the Euclidian norm.
    """

    def __init__(self, A, **kwargs):
        transposed = kwargs.get('transposed', False)
        nargout, nargin = A.shape
        if transposed:
            LinearOperator.__init__(self, nargout, nargout, **kwargs)
        else:
            LinearOperator.__init__(self, nargin, nargin, **kwargs)
        self.transposed = transposed
        if isinstance(A, LinearOperator):
            self.A = A
        else:
            self.A = PysparseLinearOperator(A, transposed=False)
        self.symmetric = True
        if self.transposed:
            self.__mul__ = self._rmul
        else:
            self.__mul__ = self._mul
        if self.logger is not None:
            self.logger.info('New squared operator with shape '+str(self.shape))
        self.T = self


    def _mul(self, x):
        self.nMatvec += 1
        return self.A.T * (self.A * x)


    def _rmul(self, x):
        self.nMatvecTransp += 1
        return self.A * (self.A.T * x)


class ReducedLinearOperator:
    """
    Given a linear operator A, implement the linear operator equivalent of
    the matrix notation A[I,J] where I and J and index sets of rows and
    columns, respectively.
    """

    def __init__(self, A, row_indices, col_indices, **kwargs):
        self.op = A             # A linear operator.
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.shape = (len(row_indices), len(col_indices))
        self.symmetric = False  # Generally.

    def __mul__(self, x):
        # Return the result of A[I,J]*x. Note that the input x must have
        # as many components as there are indices in J. The result
        # has as many components as there are indices in I.
        m, n = self.op.shape    # Shape of non-reduced operator.
        z = np.zeros(n) ; z[self.col_indices] = x[:]
        y = self.op * z
        return y[self.row_indices]


class SymmetricallyReducedLinearOperator(ReducedLinearOperator):
    def __init__(self, A, row_indices, **kwargs):
        ReducedLinearOperator.__init__(self, A, row_indices, row_indices, **kwargs)
        self.symmetric = self.op.symmetric


if __name__ == '__main__':
    from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp
    from nlpy.model import AmplModel
    from nlpy.optimize.solvers.lsqr import LSQRFramework
    import numpy as np
    import sys

    np.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

    nlp = AmplModel(sys.argv[1])
    J = sp(matrix=nlp.jac(nlp.x0))
    #J = nlp.jac(nlp.x0)
    e1 = np.ones(J.shape[0])
    e2 = np.ones(J.shape[1])

    #print 'Explicitly:'
    #print 'J*e2 = ', J*e2
    #print "J'*e1 = ", e1*J

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
    print 'Testing SimpleLinearOperator:'
    op = SimpleLinearOperator(J.shape[1], J.shape[0],
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
    print 'Solving a constrained least-squares problem with LSQR:'
    lsqr = LSQRFramework(op)
    lsqr.solve(np.random.random(nlp.m), show=True)
    print
    print 'Building a SquaredLinearOperator:'
    op2 = SquaredLinearOperator(J, log=True)
    print 'op2 * e2 = ', op2 * e2
    print 'op.T * (op * e2) = ', op.T * (op * e2)
    op3 = SquaredLinearOperator(J, transposed=True, log=True)
    print 'op3 * e1 = ', op3 * e1
    print 'op * (op.T * e1) = ', op * (op.T * e1)
    print 'op3 is symmetric: ', op3.check_symmetric()
