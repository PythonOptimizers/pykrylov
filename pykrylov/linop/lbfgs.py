# -*- coding: utf-8 -*-
"""Limited-Memory BFGS Operators.

Linear operators to represent limited-memory BFGS matrices
and their inverses.
"""

from pykrylov.linop import LinearOperator
import numpy as np

__docformat__ = 'restructuredtext'


class InverseLBFGSOperator(LinearOperator):
    """Store and manipulate inverse L-BFGS approximations.

    InverseLBFGSOperator may be used, e.g., in a LBFGS solver for
    unconstrained minimization or as a preconditioner. The limited-memory
    matrix that is implicitly stored is a positive definite approximation to
    the inverse Hessian. Therefore, search directions may be obtained by
    computing matrix-vector products only. Such products are efficiently
    computed by means of a two-loop recursion.
    """

    def __init__(self, n, npairs=5, **kwargs):
        """InverseLBFGSOperator with `npairs` {s,y} pairs in `n` variables.

        InverseLBFGSOperator(n)

        where n is the number of variables of the problem.

        :keywords:

            :npairs:     the number of {s,y} pairs stored (default: 5)
            :scaling:    enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is s'y/y'y (default: False).
        """
        # Mandatory arguments
        self.n = n
        self._npairs = npairs

        # Optional arguments
        self.scaling = kwargs.pop('scaling', False)

        # insert to points to the location where the *next* (s,y) pair
        # is to be inserted in self.s and self.y.
        self.insert = 0

        # Threshold on dot product s'y to accept a new pair (s,y).
        self.accept_threshold = 1.0e-20

        # Storage of the (s,y) pairs
        self.s = np.empty((self.n, self.npairs), 'd')
        self.y = np.empty((self.n, self.npairs), 'd')

        self.alpha = np.empty(self.npairs, 'd')    # multipliers
        self.ys = [None] * self.npairs             # dot products si'yi
        self.gamma = 1.0

        super(InverseLBFGSOperator, self).__init__(n, n,
                                                   matvec=self.lbfgs_matvec,
                                                   symmetric=True, **kwargs)

    @property
    def npairs(self):
        """Return the maximum number of {s,y} pairs stored."""
        return self._npairs

    def store(self, new_s, new_y):
        """Store the new pair {new_s,new_y}.

        A new pair is only accepted if the dot product <new_s, new_y> is over
        the threshold `self.accept_threshold`. The oldest pair is discarded in
        case the storage limit has been reached.
        """
        ys = np.dot(new_s, new_y)
        if ys <= self.accept_threshold:
            self.logger.debug('Rejecting (s,y) pair')
            return

        insert = self.insert
        self.s[:, insert] = new_s.copy()
        self.y[:, insert] = new_y.copy()
        self.ys[insert] = ys
        self.insert += 1
        self.insert = self.insert % self.npairs

    def restart(self):
        """Restart the approximation by clearing all data on past updates."""
        self.ys = [None] * self.npairs
        self.s = np.empty((self.n, self.npairs), 'd')
        self.y = np.empty((self.n, self.npairs), 'd')
        self.insert = 0
        return

    def lbfgs_matvec(self, v):
        """Compute matrix-vector product with inverse L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the inverse Hessian matrix and the
        vector v using the LBFGS two-loop recursion formula.
        """
        q = v.copy()
        s = self.s
        y = self.y
        ys = self.ys
        alpha = self.alpha
        for i in range(self.npairs):
            k = (self.insert - 1 - i) % self.npairs
            if ys[k] is not None:
                alpha[k] = np.dot(s[:, k], q) / ys[k]
                q -= alpha[k] * y[:, k]

        r = q
        if self.scaling:
            last = (self.insert - 1) % self.npairs
            if ys[last] is not None:
                self.gamma = ys[last] / np.dot(y[:, last], y[:, last])
                r *= self.gamma

        for i in range(self.npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                beta = np.dot(y[:, k], r) / ys[k]
                r += (alpha[k] - beta) * s[:, k]
        return r


class LBFGSOperator(InverseLBFGSOperator):
    """Store and manipulate forward L-BFGS approximations.

    LBFGSOperator is similar to InverseLBFGSOperator, except that
    an approximation to the direct Hessian, not its inverse, is maintained.

    This form is useful in trust region methods, where the approximate Hessian
    is used in the model problem.
    """

    def lbfgs_matvec(self, v):
        """Compute matrix-vector product with forward L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the direct Hessian matrix and the
        vector v using the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        q = v.copy()
        s = self.s
        y = self.y
        ys = self.ys
        b = np.zeros((self.n, self.npairs))
        a = np.zeros((self.n, self.npairs))

        # B = Σ aa' - bb'.
        for i in range(self.npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                b[:, k] = y[:, k] / ys[k]**.5
                bv = np.dot(b[:, k], v[:])
                q += bv * b[:, k]
                a[:, k] = s[:, k].copy()
                for j in range(i):
                    l = (self.insert + j) % self.npairs
                    if ys[l] is not None:
                        a[:, k] += np.dot(b[:, l], s[:, k]) * b[:, l]
                        a[:, k] -= np.dot(a[:, l], s[:, k]) * a[:, l]
                a[:, k] /= np.dot(s[:, k], a[:, k])**.5
                q -= np.dot(np.outer(a[:, k], a[:, k]), v[:])

        return q


class CompactLBFGSOperator(InverseLBFGSOperator):
    """Store and manipulate forward L-BFGS approximations in compact form.

    LBFGSOperator is similar bto InverseLBFGSOperator, except that it operates
    on the Hessian approximation directly, rather than the inverse.
    The so-called compact representation is used to compute this approximation
    efficiently.

    This form is useful in trust region methods, where the approximate Hessian
    is used in the model problem.
    """

    def lbfgs_matvec(self, v):
        """Compute matrix-vector product with forward L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the direct Hessian matrix and the
        vector v using the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and caching key dot products.
        """
        q = v.copy()
        r = v.copy()
        s = self.s
        y = self.y
        ys = self.ys
        prodn = 2 * self.npairs
        a = np.zeros(prodn)
        minimat = np.zeros([prodn, prodn])

        if self.scaling:
            last = (self.insert - 1) % self.npairs
            if ys[last] is not None:
                self.gamma = ys[last] / np.dot(y[:, last], y[:, last])
                r /= self.gamma

        paircount = 0
        for i in range(self.npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                a[paircount] = np.dot(r[:], s[:, k])
                paircount += 1

        j = 0
        for i in range(self.npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                a[paircount + j] = np.dot(q[:], y[:, k])
                j += 1

        # Populate small matrix to be inverted
        k_ind = 0
        for i in range(self.npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                minimat[paircount + k_ind, paircount + k_ind] = -ys[k]
                minimat[k_ind, k_ind] = np.dot(s[:, k], s[:, k]) / self.gamma
                l_ind = 0
                for j in range(i):
                    l = (self.insert + j) % self.npairs
                    if ys[l] is not None:
                        minimat[k_ind, paircount + l_ind] = np.dot(s[:, k], y[:, l])
                        minimat[paircount + l_ind, k_ind] = minimat[k_ind, paircount + l_ind]
                        minimat[k_ind, l_ind] = np.dot(s[:, k], s[:, l]) / self.gamma
                        minimat[l_ind, k_ind] = minimat[k_ind, l_ind]
                        l_ind += 1
                k_ind += 1

        if paircount > 0:
            rng = 2 * paircount
            b = np.linalg.solve(minimat[0:rng, 0:rng], a[0:rng])

        for i in range(paircount):
            k = (self.insert - paircount + i) % self.npairs
            r -= (b[i] / self.gamma) * s[:, k]
            r -= b[i + paircount] * y[:, k]

        return r


class StructuredLBFGSOperator(InverseLBFGSOperator):
    """Store and manipulate structured forward L-BFGS approximations.

    For this procedure see [Nocedal06].
    """

    def __init__(self, n, npairs=5, **kwargs):
        """InverseLBFGSOperator with `npairs` {s,y} pairs in `n` variables.

        StructuredLBFGSOperator(n)

        where n is the number of variables of the problem.

        :keywords:

            :npairs:     the number of {s,y} pairs stored (default: 5)
            :scaling:    enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is s'y/y'y (default: False).
        """
        super(StructuredLBFGSOperator, self).__init__(self, n, npairs, **kwargs)
        self.yd = np.empty((self.n, self.npairs))
        self.accept_threshold = 1e-8

    def lbfgs_matvec(self, v):
        """Compute matrix-vector product with forward L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        q = v.copy()
        s = self.s
        y = self.y
        yd = self.yd
        ys = self.ys
        npairs = self.npairs
        a = np.zeros([self.n, npairs])
        ad = np.zeros([self.n, npairs])

        aTs = np.zeros([npairs, 1])
        adTs = np.zeros([npairs, 1])

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last] / np.dot(y[:, last], y[:, last])
                q /= self.gamma

        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                coef = (self.gamma * ys[k] / np.dot(s[:, k], s[:, k]))**0.5
                a[:, k] = y[:, k] + coef * s[:, k] / self.gamma
                ad[:, k] = yd[:, k] - s[:, k] / self.gamma
                for j in range(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        alTs = np.dot(a[:, l], s[:, k]) / aTs[l]
                        adlTs = np.dot(ad[:, l], s[:, k])
                        update = alTs / aTs[l] * ad[:, l] + adlTs / aTs[l] * a[:, l] - adTs[l] / aTs[l] * alTs * a[:, l]
                        a[:, k] += coef * update
                        ad[:, k] -= update
                aTs[k] = np.dot(a[:, k], s[:, k])
                adTs[k] = np.dot(ad[:, k], s[:, k])
                aTv = np.dot(a[:, k], v[:])
                adTv = np.dot(ad[:, k], v[:])
                q += aTv / aTs[k] * ad[:, k] + adTv / aTs[k] * a[:, k] - aTv * adTs[k] / aTs[k]**2 * a[:, k]
        return q

    def store(self, new_s, new_y, new_yd):
        """Store the new pair {new_s, new_y, new_yd}.

        A new pair is only accepted if
        | y_k' s_k + (y's s_k' B_k s_k)**.5 | >= 1e-8.
        """
        ys = np.dot(new_s, new_y)
        Bs = self.matvec(new_s)
        ypBs = ys + (ys * np.dot(new_s, Bs))**0.5

        if ypBs >= self.accept_threshold:
            insert = self.insert
            self.s[:, insert] = new_s.copy()
            self.y[:, insert] = new_y.copy()
            self.yd[:, insert] = new_yd.copy()
            self.ys[insert] = ys
            self.insert += 1
            self.insert = self.insert % self.npairs
        else:
            self.log.debug('Rejecting (s,y) pair')
        return
