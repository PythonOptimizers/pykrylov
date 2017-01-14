# -*- coding: utf-8 -*-
"""Limited-Memory BFGS Operators.

Linear operators to represent limited-memory BFGS matrices and their inverses.
"""

from pykrylov.linop import LQNLinearOperator, StructuredLQNLinearOperator
import numpy as np

__docformat__ = 'restructuredtext'


class InverseLBFGSOperator(LQNLinearOperator):
    """Store and manipulate inverse L-BFGS approximations.

    :class: `InverseLBFGSOperator` may be used, e.g., in a L-BFGS solver for
    unconstrained minimization or as a preconditioner. The limited-memory
    matrix that is implicitly stored is a positive definite approximation to
    the inverse Hessian. Therefore, search directions may be obtained by
    computing matrix-vector products only. Such products are efficiently
    computed by means of a two-loop recursion.
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `InverseLBFGSOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(InverseLBFGSOperator, self).__init__(n, npairs, **kwargs)

    def _storing_test(self, new_s, new_y, ys):
        u"""Test if new pair {s, y} is to be stored.

        A new pair is only accepted if the dot product yᵀs is over the
        threshold `self.accept_threshold`. The oldest pair is discarded in case
        the storage limit has been reached.
        """
        return ys > self.accept_threshold

    def qn_matvec(self, v):
        """Compute matrix-vector product with inverse L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the inverse Hessian matrix and the
        vector v using the L-BFGS two-loop recursion formula.
        """
        self.n_matvec += 1
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

    :class: `LBFGSOperator` is similar to :class: `InverseLBFGSOperator`,
    except that an approximation to the direct Hessian, not its inverse, is
    maintained.

    This form is useful in trust region methods, where the approximate Hessian
    is used in the model problem.
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `LBFGSOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(LBFGSOperator, self).__init__(n, npairs, **kwargs)

    def qn_matvec(self, v):
        """Compute matrix-vector product with forward L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the direct Hessian matrix and the
        vector v using the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        self.n_matvec += 1
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

    :class: `CompactLBFGSOperator` is similar to :class:
    `InverseLBFGSOperator`, except that it operates on the Hessian
    approximation directly, rather than the inverse. The so-called compact
    representation is used to compute this approximation efficiently.

    This form is useful in trust region methods, where the approximate Hessian
    is used in the model problem.
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `CompactLBFGSOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(CompactLBFGSOperator, self).__init__(n, npairs, **kwargs)

    def qn_matvec(self, v):
        """Compute matrix-vector product with forward L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the direct Hessian matrix and the
        vector v using the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and caching key dot products.
        """
        self.n_matvec += 1

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
                        minimat[k_ind, paircount + l_ind] = np.dot(s[:, k],
                                                                   y[:, l])
                        minimat[paircount + l_ind, k_ind] = minimat[k_ind,
                                                                    (paircount +
                                                                     l_ind)]
                        minimat[k_ind, l_ind] = np.dot(s[:, k],
                                                       s[:, l]) / self.gamma
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


class StructuredLBFGSOperator(StructuredLQNLinearOperator):
    """Store and manipulate structured forward L-BFGS approximations.

    For this procedure see[Nocedal06].
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `StructuredLBFGSOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s,y, yd} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(StructuredLBFGSOperator, self).__init__(n, npairs, **kwargs)
        self.accept_threshold = 1e-10

    def _storing_test(self, new_s, new_y, new_yd, ys):
        u"""Test if new pair {s, y, yd} is to be stored.

        A new pair {s, y, yd} is only accepted if

            ∣yᵀs + √(yᵀs sᵀBs)∣ ⩾ self.accept_threshold
        """
        Bs = self.qn_matvec(new_s)
        sBs = np.dot(new_s, Bs)

        # Supress python runtime warnings
        if ys < 0.0 or sBs < 0.0:
            return False

        ypBs = ys + (ys * sBs)**0.5
        return ypBs >= self.accept_threshold

    def qn_matvec(self, v):
        """Compute matrix-vector product with forward L-BFGS approximation.

        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        self.n_matvec += 1
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
                # Form a[] and ad[] vectors for current step
                ad[:, k] = yd[:, k] - s[:, k] / self.gamma
                Bsk = s[:, k] / self.gamma
                for j in range(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        aTsk = np.dot(a[:, l], s[:, k])
                        adTsk = np.dot(ad[:, l], s[:, k])
                        aTsl = np.dot(a[:, l], s[:, l])
                        adTsl = np.dot(ad[:, l], s[:, l])
                        update = (aTsk / aTsl) * ad[:, l] + (adTsk / aTsl) * a[:, l] - \
                            (aTsk * adTsl / aTsl**2) * a[:, l]
                        Bsk += update.copy()
                        ad[:, k] -= update.copy()
                a[:, k] = y[:, k] + (ys[k] / np.dot(s[:, k], Bsk))**0.5 * Bsk

                # Form inner products with current s[] and input vector
                aTs[k] = np.dot(a[:, k], s[:, k])
                adTs[k] = np.dot(ad[:, k], s[:, k])
                aTv = np.dot(a[:, k], v[:])
                adTv = np.dot(ad[:, k], v[:])

                q += (aTv / aTs[k]) * ad[:, k] + (adTv / aTs[k]) * a[:, k] - \
                    (aTv * adTs[k] / aTs[k]**2) * a[:, k]

        return q
