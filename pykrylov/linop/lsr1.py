# -*- coding: utf-8 -*-
"""Limited-Memory SR1 (Symmetric Rank 1) Operators.

Linear operators to represent limited-memory SR1 matrices and their inverses.
L-SR1 matrices may not be positive-definite.
"""
from pykrylov.linop import LQNLinearOperator, StructuredLQNLinearOperator

import numpy as np
from numpy.linalg import norm


class LSR1Operator(LQNLinearOperator):
    """Store and manipulate forward L-SR1 approximations.

    LSR1Operator may be used, e.g., in trust region methods, where the
    approximate Hessian is used in the model problem. L-SR1 has the advantage
    over L-BFGS and L-DFP of permitting approximations that are not
    positive-definite.
    LSR1 quasi-Newton using an unrolling formula.
    For this procedure see [Nocedal06].
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `LSR1Operator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(LSR1Operator, self).__init__(n, npairs, **kwargs)
        self.accept_threshold = 1e-8

    def _storing_test(self, new_s, new_y, ys):
        u"""Test if new pair {s, y} is to be stored.

        A new pair {s, y} is only accepted if

            ∣sᵀ(y - B s)∣ ⩾ 1e-8 ‖s‖ ‖y - B s‖.
        """
        Bs = self.qn_matvec(new_s)
        ymBs = new_y - Bs
        criterion = abs(np.dot(ymBs, new_s)) >= self.accept_threshold * \
            norm(new_s) * norm(ymBs)
        ymBsTs_criterion = abs(np.dot(ymBs, new_s)) >= 1e-15
        ys = np.dot(new_s, new_y)

        ys_criterion = True
        scaling_criterion = True
        yms_criterion = True
        if self.scaling:
            if abs(ys) >= 1e-15:
                scaling_factor = ys / np.dot(new_y, new_y)
                scaling_criterion = norm(new_y -
                                         new_s / scaling_factor) >= 1e-10
            else:
                ys_criterion = False
        else:
            if norm(new_y - new_s) < 1e-10:
                yms_criterion = False

        return (ymBsTs_criterion and yms_criterion and scaling_criterion and
                criterion and ys_criterion)

    def qn_matvec(self, v):
        """Compute matrix-vector product with L-SR1 approximation.

        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using the
        unrolling formula.
        """
        self.n_matvec += 1

        q = v.copy()
        s = self.s
        y = self.y
        ys = self.ys
        npairs = self.npairs
        a = np.zeros([self.n, npairs])
        aTs = np.zeros([npairs, 1])

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last] / np.dot(y[:, last], y[:, last])
                q /= self.gamma

        for i in xrange(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                a[:, k] = y[:, k] - s[:, k] / self.gamma
                for j in xrange(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        a[:, k] -= np.dot(a[:, l], s[:, k]) / aTs[l] * a[:, l]
                aTs[k] = np.dot(a[:, k], s[:, k])
                q += np.dot(a[:, k], v[:]) / aTs[k] * a[:, k]
        return q


class CompactLSR1Operator(LSR1Operator):
    """Store and manipulate forward L-SR1 approximations.

    The so-called compact representation is used to compute this approximation
    efficiently.
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `CompactLSR1Operator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(CompactLSR1Operator, self).__init__(n, npairs, **kwargs)
        self.accept_threshold = 1.0e-8

    def qn_matvec(self, v):
        """Compute matrix-vector product with L-SR1 approximation.

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
        ys = self.ys
        npairs = self.npairs
        a = np.zeros([npairs, 1], 'd')
        minimat = np.zeros([npairs, npairs], 'd')

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last] / np.dot(y[:, last], y[:, last])
                q /= self.gamma

        paircount = 0
        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                a[k] = np.dot(y[:, k], v[:]) - np.dot(s[:, k], q[:])
                paircount += 1

        # Populate small matrix to be inverted
        k_ind = 0
        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                minimat[k, k] = ys[k] - np.dot(s[:, k], s[:, k]) / self.gamma
                l_ind = 0
                for j in range(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        minimat[k, l] = np.dot(s[:, k], y[:, l]) - \
                            np.dot(s[:, k], s[:, l]) / self.gamma
                        minimat[l, k] = minimat[k, l]
                        l_ind += 1
                k_ind += 1

        if paircount > 0:
            rng = paircount
            b = np.linalg.solve(minimat[0:rng, 0:rng], a[0:rng])

        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                q += b[k] * y[:, k] - b[k] / self.gamma * s[:, k]
        return q


class InverseLSR1Operator(LSR1Operator):
    """Store and manipulate inverse L-SR1 approximations."""

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `InverseLSR1Operator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(InverseLSR1Operator, self).__init__(n, npairs, **kwargs)
        self.accept_threshold = 1e-8

    def qn_matvec(self, v):
        """Compute matrix-vector product with inverse L-SR1 approximation.

        Compute a matrix-vector product between the current limited-memory
        approximation to the inverse Hessian matrix and the vector v using
        the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        self.n_matvec += 1

        q = v.copy()
        s = self.s
        y = self.y
        ys = self.ys
        npairs = self.npairs
        a = np.zeros(npairs, 'd')
        minimat = np.zeros([npairs, npairs], 'd')

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last] / np.dot(y[:, last], y[:, last])
                q *= self.gamma

        paircount = 0
        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                a[k] = np.dot(s[:, k], v[:]) - np.dot(y[:, k], q[:])
                paircount += 1

        # Populate small matrix to be inverted
        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                minimat[k, k] = ys[k] - np.dot(y[:, k], y[:, k]) * self.gamma
                for j in range(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        minimat[k, l] = np.dot(y[:, k], s[:, l]) - \
                            np.dot(y[:, k], y[:, l]) * self.gamma
                        minimat[l, k] = minimat[k, l]

        if paircount > 0:
            rng = paircount
            b = np.linalg.solve(minimat[0:rng, 0:rng], a[0:rng])

        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                q += b[k] * s[:, k]
                q -= (b[k] * self.gamma) * y[:, k]

        return q


class StructuredLSR1Operator(StructuredLQNLinearOperator):
    """Store and manipulate structured forward L-SR1 approximations.

    Structured L-SR1 quasi-Newton approximation using an unrolling formula.
    For this procedure see [Nocedal06].
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `StructuredLSR1Operator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y, yd} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(StructuredLSR1Operator, self).__init__(n, npairs,
                                                     **kwargs)
        self.accept_threshold = 1e-8

    def _storing_test(self, new_s, new_y, new_yd, ys):
        u"""Test if new pair {s, y, yd} is to be stored.

        A new pair {s, y, yd} is only accepted if

            ∣sᵀ(yd - B s)∣ ⩾ 1e-8 ‖s‖ ‖yd - B s‖.
        """
        Bs = self.qn_matvec(new_s)
        ymBs = new_yd - Bs
        criterion = abs(np.dot(ymBs, new_s)) >= self.accept_threshold * \
            norm(new_s) * norm(ymBs)
        ymBsTs_criterion = abs(np.dot(ymBs, new_s)) >= 1e-15
        ys = np.dot(new_s, new_y)

        ys_criterion = True
        scaling_criterion = True
        yms_criterion = True
        if self.scaling:
            if abs(ys) >= 1e-15:
                scaling_factor = ys / np.dot(new_y, new_y)
                scaling_criterion = norm(new_y -
                                         new_s / scaling_factor) >= 1e-10
            else:
                ys_criterion = False
        else:
            if norm(new_y - new_s) < 1e-10:
                yms_criterion = False

        return (ymBsTs_criterion and yms_criterion and scaling_criterion and
                criterion and ys_criterion)

    def qn_matvec(self, v):
        """Compute matrix-vector product with forward L-SR1 approximation.

        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the unrolling formula.
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
                # Form all a[] and ad[] vectors for the current step
                a[:, k] = y[:, k] - s[:, k] / self.gamma
                ad[:, k] = yd[:, k] - s[:, k] / self.gamma
                for j in range(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        aTsk = np.dot(a[:, l], s[:, k])
                        adTsk = np.dot(ad[:, l], s[:, k])
                        aTsl = np.dot(a[:, l], s[:, l])
                        adTsl = np.dot(ad[:, l], s[:, l])
                        update = (aTsk / aTsl) * ad[:, l] + (adTsk / aTsl) * a[:, l] - \
                            (aTsk * adTsl / aTsl**2) * a[:, l]
                        a[:, k] -= update.copy()
                        ad[:, k] -= update.copy()

                # Form inner products with current s[] and input vector
                aTs[k] = np.dot(a[:, k], s[:, k])
                adTs[k] = np.dot(ad[:, k], s[:, k])
                aTv = np.dot(a[:, k], v[:])
                adTv = np.dot(ad[:, k], v[:])

                q += (aTv / aTs[k]) * ad[:, k] + (adTv / aTs[k]) * a[:, k] - \
                    (aTv * adTs[k] / aTs[k]**2) * a[:, k]
        return q
