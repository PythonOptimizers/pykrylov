from pykrylov.linop import BaseLinearOperator, ShapeError, null_log
import numpy as np


class BlockLinearOperator(BaseLinearOperator):
    """
    A linear operator defined by blocks. Each block must be a linear operator.
    """

    def __init__(self, blocks, symmetric=False, **kwargs):
        """
        `blocks` should be a list of lists describing the blocks row-wise.
        If there is only one block row, it should be specified as
        `[[b1, b2, ..., bn]]`, not as `[b1, b2, ..., bn]`.

        If the overall linear operator is symmetric, only its upper triangle
        need be specified, e.g., `[[A,B,C], [D,E], [F]]`.
        """

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
                for j in range(i):
                    self._blocks[i].insert(0, self._blocks[j][i].T)

        else:
            self._blocks = blocks

        log = kwargs.get('logger', null_log)
        log.debug('Building new BlockLinearOperator')

        self.transposed = kwargs.pop('transposed', False)

        nargins = [[blk.shape[-1] for blk in row] for row in blocks]
        log.debug('nargins = ' + repr(nargins))
        nargins_by_row = [nargin[0] for nargin in nargins]
        if min(nargins_by_row) != max(nargins_by_row):
            raise ShapeError('Inconsistent block shapes')

        nargouts = [[blk.shape[0] for blk in row] for row in blocks]
        log.debug('nargouts = ' + repr(nargouts))
        for row in nargouts:
            if min(row) != max(row):
                raise ShapeError('Inconsistent block shapes')

        # Keep track of the segmentation for easier reference.
        self.nargins = nargins[0]
        self.nargouts = [nargout[0] for nargout in nargouts]

        log.debug('self.nargins = ' + repr(self.nargins))
        log.debug('self.nargouts = ' + repr(self.nargouts))

        nargin = sum(nargins[0])
        nargout = sum([out[0] for out in nargouts])

        transpose_of = kwargs.pop('transpose_of', None)

        super(BlockLinearOperator, self).__init__(nargin=nargin,
                                                  nargout=nargout,
                                                  symmetric=symmetric,
                                                  **kwargs)

        # Define transpose operator.
        if symmetric:
            self.T = self
        else:
            if transpose_of is None:
                # Create 'pointer' to transpose operator.
                blocksT = map(lambda *row: [blk.T for blk in row], *blocks)
                self.T = BlockLinearOperator(blocksT,
                                             symmetric,
                                             transposed=not self.transposed,
                                             transpose_of=self,
                                             **kwargs)
            else:
                # Use operator supplied as transpose operator.
                if isinstance(transpose_of, BaseLinearOperator):
                    self.T = transpose_of
                else:
                    msg = 'kwarg transposed_of must be a BaseLinearOperator.'
                    msg += ' Got ' + str(transpose_of.__class__)
                    raise ValueError(msg)

    def __mul__(self, x):
        nx = len(x)
        self.logger.debug('Multiplying with a vector of size %d' % nx)
        self.logger.debug('nargin=%d, nargout=%d' % (self.nargin, self.nargout))
        if nx != self.nargin:
            raise ShapeError('Multiplying with vector of wrong shape.')
        y = np.zeros(self.nargout)

        nblk_row = len(self._blocks)
        nblk_col = len(self._blocks[0])

        row_start = col_start = 0
        for row in range(nblk_row):
            row_end = row_start + self.nargouts[row]
            yout = y[row_start:row_end]
            for col in range(nblk_col):
                col_end = col_start + self.nargins[col]
                xin = x[col_start:col_end]
                B = self._blocks[row][col]
                yout[:] += B * xin
                col_start = col_end
            row_start = row_end
            col_start = 0

        return y

    def __getitem__(self, idx):
        return self._blocks[idx[0]][idx[1]]

    def __getslice__(self, slice):
        # A priori, a slice is not symmetric.
        return BlockLinearOperator(self._blocks[slice], symmetric=False)


class BlockDiagonalLinearOperator(BaseLinearOperator):
    """
    A block diagonal linear operator. Each block must be a linear operator.
    """

    def __init__(self, blocks, symmetric=False, **kwargs):
        """
        The blocks may be specified as one list, e.g., `[A, B, C]`.
        """

        self._blocks = blocks
        self.nargins = [blk.shape[-1] for blk in blocks]
        self.nargouts = [blk.shape[0] for blk in blocks]

        nargin = sum(self.nargins)
        nargout = sum(self.nargouts)

        self.transposed = kwargs.pop('transposed', False)
        transpose_of = kwargs.pop('transpose_of', None)

        super(BlockDiagonalLinearOperator, self).__init__(nargin=nargin,
                                                          nargout=nargout,
                                                          symmetric=symmetric,
                                                          **kwargs)

        # Define transpose operator.
        if symmetric:
            self.T = self
        else:
            if transpose_of is None:
                # Create 'pointer' to transpose operator.
                blocksT = [blk.T for blk in blocks]
                self.T = BlockDiagonalLinearOperator(blocksT,
                                                symmetric,
                                                transposed=not self.transposed,
                                                transpose_of=self,
                                                **kwargs)
            else:
                # Use operator supplied as transpose operator.
                if isinstance(transpose_of, BaseLinearOperator):
                    self.T = transpose_of
                else:
                    msg = 'kwarg transposed_of must be a LinearOperator.'
                    msg += ' Got ' + str(transpose_of.__class__)
                    raise ValueError(msg)

    def __mul__(self, x):
        nx = len(x)
        self.logger.debug('Multiplying with a vector of size %d' % nx)
        self.logger.debug('nargin=%d, nargout=%d' % (self.nargin, self.nargout))
        if nx != self.nargin:
            raise ShapeError('Multiplying with vector of wrong shape.')
        y = np.empty(self.nargout)

        nblks = len(self._blocks)

        row_start = col_start = 0
        for blk in range(nblks):
            row_end = row_start + self.nargouts[blk]
            yout = y[row_start:row_end]

            col_end = col_start + self.nargins[blk]
            xin = x[col_start:col_end]

            B = self._blocks[blk]
            yout[:] = B * xin

            col_start = col_end
            row_start = row_end

        return y

    def __getitem__(self, idx):
        return self._blocks[idx]


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

    A = LinearOperator(nargin=3, nargout=3, matvec=lambda v: 2*v, symmetric=True)
    B = LinearOperator(nargin=4, nargout=3, matvec=lambda v: v[:3],
                       matvec_transp=lambda v: np.concatenate((v, np.zeros(1))))
    C = LinearOperator(nargin=3, nargout=2, matvec=lambda v: v[:2],
                       matvec_transp=lambda v: np.concatenate((v, np.zeros(1))))
    D = LinearOperator(nargin=4, nargout=2, matvec=lambda v: v[:2],
                       matvec_transp=lambda v: np.concatenate((v, np.zeros(2))))
    E = LinearOperator(nargin=4, nargout=4, matvec=lambda v: -v, symmetric=True)

    print A.shape, A.T.shape
    print B.shape, B.T.shape
    print C.shape, C.T.shape
    print D.shape, D.T.shape
    print E.shape, E.T.shape

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
