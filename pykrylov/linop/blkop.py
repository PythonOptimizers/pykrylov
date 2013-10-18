from pykrylov.linop import BaseLinearOperator, LinearOperator
from pykrylov.linop import ShapeError, null_log
import numpy as np
import itertools


class BlockLinearOperator(LinearOperator):
    """
    A linear operator defined by blocks. Each block must be a linear operator.

    `blocks` should be a list of lists describing the blocks row-wise.
    If there is only one block row, it should be specified as
    `[[b1, b2, ..., bn]]`, not as `[b1, b2, ..., bn]`.

    If the overall linear operator is symmetric, only its upper triangle
    need be specified, e.g., `[[A,B,C], [D,E], [F]]`, and the blocks on the
    diagonal must be square and symmetric.
    """

    def __init__(self, blocks, symmetric=False, **kwargs):
        # If building a symmetric operator, fill in the blanks.
        # They're just references to existing objects.
        if symmetric:
            nrow = len(blocks)
            ncol = len(blocks[0])
            if nrow != ncol:
                raise ShapeError('Inconsistent shape.')

            for block_row in blocks:
                if not block_row[0].symmetric:
                    raise ValueError('Blocks on diagonal must be symmetric.')

            self._blocks = blocks[:]
            for i in range(1, nrow):
                for j in range(i-1,-1,-1):
                    self._blocks[i].insert(0, self._blocks[j][i].T)

        else:
            self._blocks = blocks

        log = kwargs.get('logger', null_log)
        log.debug('Building new BlockLinearOperator')

        nargins = [[blk.shape[-1] for blk in row] for row in self._blocks]
        log.debug('nargins = ' + repr(nargins))
        nargins_by_row = [nargin[0] for nargin in nargins]
        if min(nargins_by_row) != max(nargins_by_row):
            raise ShapeError('Inconsistent block shapes')

        nargouts = [[blk.shape[0] for blk in row] for row in self._blocks]
        log.debug('nargouts = ' + repr(nargouts))
        for row in nargouts:
            if min(row) != max(row):
                raise ShapeError('Inconsistent block shapes')

        nargin = sum(nargins[0])
        nargout = sum([out[0] for out in nargouts])

        # Create blocks of transpose operator.
        blocksT = map(lambda *row: [blk.T for blk in row], *self._blocks)

        def blk_matvec(x, blks):
            nargins = [[blk.shape[-1] for blk in blkrow] for blkrow in blks]
            nargouts = [[blk.shape[0] for blk in blkrow] for blkrow in blks]
            nargin = sum(nargins[0])
            nargout = sum([out[0] for out in nargouts])
            nx = len(x)
            self.logger.debug('Multiplying with a vector of size %d' % nx)
            self.logger.debug('nargin=%d, nargout=%d' % (nargin, nargout))
            if nx != nargin:
                raise ShapeError('Multiplying with vector of wrong shape.')

            result_type = np.result_type(self.dtype, x.dtype)
            y = np.zeros(nargout, dtype=result_type)

            nblk_row = len(blks)
            nblk_col = len(blks[0])

            row_start = col_start = 0
            for row in range(nblk_row):
                row_end = row_start + nargouts[row][0]
                yout = y[row_start:row_end]
                for col in range(nblk_col):
                    col_end = col_start + nargins[0][col]
                    xin = x[col_start:col_end]
                    B = blks[row][col]
                    yout[:] += B * xin
                    col_start = col_end
                row_start = row_end
                col_start = 0

            return y

        flat_blocks = list(itertools.chain(*blocks))
        blk_dtypes = [blk.dtype for blk in flat_blocks]
        op_dtype = np.result_type(*blk_dtypes)

        super(BlockLinearOperator, self).__init__(nargin, nargout,
                                symmetric=symmetric,
                                matvec=lambda x: blk_matvec(x, self._blocks),
                                matvec_transp=lambda x: blk_matvec(x, blocksT),
                                dtype=op_dtype)

        self.T._blocks = blocksT

    @property
    def blocks(self):
        "The list of blocks defining the block operator."
        return self._blocks

    def __getitem__(self, indices):
        blks = np.matrix(self._blocks, dtype=object)[indices]
        # If indexing narrowed it down to a single block, return it.
        if isinstance(blks, BaseLinearOperator):
            return blks
        # Otherwise, we have a matrix of blocks.
        return BlockLinearOperator(blks.tolist(), symmetric=False)

    def __contains__(self, op):
        return op in self._blocks

    def __iter__(self):
        for block in self._blocks:
            yield block


class BlockDiagonalLinearOperator(LinearOperator):
    """
    A block diagonal linear operator. Each block must be a linear operator.
    The blocks may be specified as one list, e.g., `[A, B, C]`.
    """

    def __init__(self, blocks, **kwargs):

        symmetric = reduce(lambda x,y: x and y,
                           [blk.symmetric for blk in blocks])

        self._blocks = blocks

        log = kwargs.get('logger', null_log)
        log.debug('Building new BlockDiagonalLinearOperator')

        nargins = [blk.shape[-1] for blk in blocks]
        log.debug('nargins = ' + repr(nargins))

        nargouts = [blk.shape[0] for blk in blocks]
        log.debug('nargouts = ' + repr(nargouts))

        nargin = sum(nargins)
        nargout = sum(nargouts)

        # Create blocks of transpose operator.
        blocksT = [blk.T for blk in blocks]

        def blk_matvec(x, blks):
            nx = len(x)
            nargins = [blk.shape[-1] for blk in blocks]
            nargin = sum(nargins)
            nargouts = [blk.shape[0] for blk in blocks]
            nargout = sum(nargouts)
            self.logger.debug('Multiplying with a vector of size %d' % nx)
            self.logger.debug('nargin=%d, nargout=%d' % (nargin, nargout))
            if nx != nargin:
                raise ShapeError('Multiplying with vector of wrong shape.')

            result_type = np.result_type(self.dtype, x.dtype)
            y = np.empty(nargout, dtype=result_type)

            nblks = len(blks)

            row_start = col_start = 0
            for blk in range(nblks):
                row_end = row_start + nargouts[blk]
                yout = y[row_start:row_end]

                col_end = col_start + nargins[blk]
                xin = x[col_start:col_end]

                B = blks[blk]
                yout[:] = B * xin

                col_start = col_end
                row_start = row_end

            return y

        blk_dtypes = [blk.dtype for blk in blocks]
        op_dtype = np.result_type(*blk_dtypes)

        super(BlockDiagonalLinearOperator, self).__init__(nargin, nargout,
                                symmetric=symmetric,
                                matvec=lambda x: blk_matvec(x, self._blocks),
                                matvec_transp=lambda x: blk_matvec(x, blocksT),
                                dtype=op_dtype)

        self.T._blocks = blocksT

    @property
    def blocks(self):
        "The list of blocks defining the block diagonal operator."
        return self._blocks

    def __getitem__(self, idx):
        blks = self._blocks[idx]
        if isinstance(idx, slice):
            return BlockDiagonalLinearOperator(blks, symmetric=self.symmetric)
        return blks

    def __setitem__(self, idx, ops):
        if not isinstance(ops, BaseLinearOperator):
            if isinstance(ops, list) or isinstance(ops, tuple):
                for op in ops:
                    if not isinstance(op, BaseLinearOperator):
                        msg  = 'Block operators can only contain'
                        msg += ' linear operators'
                        raise ValueError(msg)
        self._blocks[idx] = ops


class BlockPreconditioner(BlockLinearOperator):
    """
    An alias for ``BlockLinearOperator`` with a ``solve`` method equivalent to
    ``__mul__``.
    """

    def solve(self, x):
        return self.__call__(x)


class BlockDiagonalPreconditioner(BlockDiagonalLinearOperator):
    """
    An alias for ``BlockDiagonalLinearOperator`` with a ``solve`` method
    equivalent to ``__mul__``.
    """

    def solve(self, x):
        return self.__call__(x)


if __name__ == '__main__':

    from pykrylov.linop import LinearOperator
    import logging
    import sys

    # Create root logger.
    log = logging.getLogger('blk-ops')
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(name)-8s %(levelname)-8s %(message)s')
    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)
    log.addHandler(hndlr)

    A = LinearOperator(nargin=3, nargout=3,
                       matvec=lambda v: 2*v, symmetric=True)
    B = LinearOperator(nargin=4, nargout=3, matvec=lambda v: v[:3],
                       matvec_transp=lambda v: np.concatenate((v, np.zeros(1))))
    C = LinearOperator(nargin=3, nargout=2, matvec=lambda v: v[:2],
                       matvec_transp=lambda v: np.concatenate((v, np.zeros(1))))
    D = LinearOperator(nargin=4, nargout=2, matvec=lambda v: v[:2],
                       matvec_transp=lambda v: np.concatenate((v, np.zeros(2))))
    E = LinearOperator(nargin=4, nargout=4,
                       matvec=lambda v: -v, symmetric=True)
    E = LinearOperator(nargin=4, nargout=4,
                       matvec=lambda v: -v, symmetric=True)
    F = LinearOperator(nargin=2, nargout=2,
                       matvec=lambda v: v, symmetric=True)


    print A.shape, A.T.shape
    print B.shape, B.T.shape
    print C.shape, C.T.shape
    print D.shape, D.T.shape
    print E.shape, E.T.shape
    print F.shape, F.T.shape

    # Build [A  B].
    K1 = BlockLinearOperator([[A, B]], logger=log)

    # Build [A  B]
    #       [C  D]
    K2 = BlockLinearOperator([[A, B], [C, D]], logger=log)

    x = np.ones(K2.shape[1])
    K2x = K2 * x
    print 'K2*e = ', K2x

    y = np.ones(K2.shape[0])
    K2Ty = K2.T * y
    print 'K2.T*e = ', K2Ty

    # Build [A  B]
    #       [B' E]
    K3 = BlockLinearOperator([[A, B], [E]], symmetric=True, logger=log)
    y = np.ones(K3.shape[0])
    K3y = K3 * y
    print 'K3*e = ', K3y
    K3Ty = K3.T * y
    print 'K3.T*e = ', K3Ty

    K4 = BlockDiagonalLinearOperator([A, E], symmetric=True, logger=log)
    y = np.ones(K4.shape[0])
    K4y = K4 * y
    print 'K4*e = ', K4y
    K4Ty = K4.T * y
    print 'K4.T*e = ', K4Ty

    # Build [A  B  C']
    #       [B' E  D']
    #       [C  D  F ]
    K5 = BlockLinearOperator([[A, B, C.T], [E, D.T], [F]], symmetric=True, logger=log)
    y = np.ones(K5.shape[0])
    K5y = K5 * y
    print 'K5 = \n', K5.to_array()
    print 'K5*e = ', K5y
    K5Ty = K5.T * y
    print 'K5.T*e = ', K5Ty
