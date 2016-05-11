# -*- coding: utf-8 -*-
"""Generic Limited-Memory Quasi Newton Operators.

Linear operators to represent limited-memory quasi-Newton matrices
or their inverses.
"""

from pykrylov.linop import LinearOperator
import numpy as np

__docformat__ = 'restructuredtext'


class LQNLinearOperator(LinearOperator):
    """Store and manipulate Limited-memory Quasi-Newton approximations."""

    def __init__(self, n, npairs=5, **kwargs):
        """Instantiate a :class: `LQNLinearOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s,y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix' (default: False).
        """
        # Mandatory arguments
        self.n = n
        self._npairs = npairs

        # Optional arguments
        self.scaling = kwargs.pop('scaling', False)

        # insert to points to the location where the *next* {s, y} pair
        # is to be inserted in self.s and self.y.
        self.insert = 0

        # Threshold on dot product s'y to accept a new pair {s, y}.
        self.accept_threshold = 1.0e-20

        # Storage of the (s,y) pairs
        self.s = np.empty((self.n, self.npairs), 'd')
        self.y = np.empty((self.n, self.npairs), 'd')

        self.alpha = np.empty(self.npairs, 'd')    # multipliers
        self.ys = [None] * self.npairs             # dot products si'yi
        self.gamma = 1.0

        # Keep track of number of matrix-vector products.
        self.n_matvec = 0

        super(LQNLinearOperator, self).__init__(n, n,
                                                matvec=self.qn_matvec,
                                                symmetric=True, **kwargs)

    @property
    def npairs(self):
        """Return the maximum number of {s,y} pairs stored."""
        return self._npairs

    def _storing_test(self, new_s, new_y, ys):
        """Test if new pair {s, y} is to be stored."""
        raise NotImplementedError("Must be subclassed.")

    def store(self, new_s, new_y):
        """Store the new pair {new_s,new_y}.

        A new pair is only accepted if `self._storing_test()` is True. The
        oldest pair is then discarded in case the storage limit has been
        reached.
        """
        ys = np.dot(new_s, new_y)

        if not self._storing_test(new_s, new_y, ys):
            self.logger.debug('Rejecting {s,y} pair')
            return

        insert = self.insert
        self.s[:, insert] = new_s.copy()
        self.y[:, insert] = new_y.copy()
        self.ys[insert] = ys
        self.insert += 1
        self.insert = self.insert % self.npairs
        return

    def restart(self):
        """Restart the approximation by clearing all data on past updates."""
        self.ys = [None] * self.npairs
        self.s = np.empty((self.n, self.npairs), 'd')
        self.y = np.empty((self.n, self.npairs), 'd')
        self.insert = 0
        return

    def qn_matvec(self, v):
        """Compute matrix-vector product."""
        raise NotImplementedError("Must be subclassed.")


class StructuredLQNLinearOperator(LQNLinearOperator):
    u"""Store and manipulate structured limited-memory Quasi-Newton approximations.

    Structured quasi-Newton approximations may be used, e.g., in augmented Lagrangian methods or in nonlinear least-squares, where Hessian has a special structure.

    If Φ(x;λ,ρ) is the augmented Lagrangian function of an equality constrained optimization problem,
        ∇ₓₓΦ(x;λ,ρ) = ∇ₓₓL(x,λ+ρc(x)) + ρJ(x)ᵀJ(x).
    The structured quasi-Newton update takes the form
        B_{k+1} := S_{k+1} + ρJᵀ J
    where S_{k+1} ≈ ∇ₓₓL(x,λ+ρc(x)).
    See [Arreckx15]_ for more details.

    [Arreckx15] A matrix-free augmented lagrangian algorithm with application to large-scale structural design optimization, S. Arreckx, A. Lambe, J. R. R. A. Martins and D. Orban, Optimization and Engineering, 2015, 1--26.
    """

    def __init__(self, n, npairs=5, **kwargs):
        """Instantiate a :class: `StructuredLQNLinearOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y, yd} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix' (default: False).
        """
        super(StructuredLQNLinearOperator, self).__init__(n, npairs, **kwargs)
        self.yd = np.empty((self.n, self.npairs))

    def store(self, new_s, new_y, new_yd):
        """Store the new pair {new_s, new_y, new_yd}.

        A new pair is only accepted if `self._storing_test()` is True. The
        oldest pair is then discarded in case the storage limit has been
        reached.
        """
        ys = np.dot(new_s, new_y)

        if not self._storing_test(new_s, new_y, new_yd, ys):
            self.logger.debug('Rejecting {s, y, yd} pair')
            return

        insert = self.insert
        self.s[:, insert] = new_s.copy()
        self.y[:, insert] = new_y.copy()
        self.yd[:, insert] = new_yd.copy()
        self.ys[insert] = ys
        self.insert += 1
        self.insert = self.insert % self.npairs
        return
