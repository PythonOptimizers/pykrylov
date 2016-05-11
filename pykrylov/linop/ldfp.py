# -*- coding: utf-8 -*-
"""Limited-Memory DFP Operators.

Linear operators to represent limited-memory DFP matrices
and their inverses.
"""
from pykrylov.linop import InverseLBFGSOperator, LBFGSOperator

__docformat__ = 'restructuredtext'


class LDFPOperator(InverseLBFGSOperator):
    """Store and manipulate forward L-DFP approximations.

    Forward L-DFP is equivalent to inverse L-BFGS where pairs {s, y} are
    switched to {y, s}.
    See the documentation of `InverseLBFGSOperator`.
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `LDFPOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(LDFPOperator, self).__init__(n, npairs, **kwargs)

    def store(self, new_s, new_y):
        """Store the new pair {new_s, new_y}."""
        # Simply swap s and y.
        super(LDFPOperator, self).store(new_y, new_s)


class InverseLDFPOperator(LBFGSOperator):
    """Store and manipulate inverse L-DFP approximations.

    Inverse L-DFP is equivalent to forward L-BFGS where pairs {s, y} are
    switched to {y, s}.
    See the documentation of :class: `LBFGSOperator`.
    """

    def __init__(self, n, npairs=5, **kwargs):
        u"""Instantiate a :class: `InverseLDFPOperator`.

        :parameters:
            :n: the number of variables.
            :npairs: the number of {s, y} pairs stored (default: 5).

        :keywords:
            :scaling: enable scaling of the 'initial matrix'. Scaling is
                         done as 'method M3' in the LBFGS paper by Zhou and
                         Nocedal; the scaling factor is sᵀy/yᵀy
                         (default: False).
        """
        super(InverseLDFPOperator, self).__init__(n, npairs, **kwargs)

    def store(self, new_s, new_y):
        """Store the new pair {new_s, new_y}."""
        # Simply swap s and y.
        super(InverseLDFPOperator, self).store(new_y, new_s)
