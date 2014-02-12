"""Test suite for the linop module."""
#Copyright (c) 2013-2014, Ghislain Vaillant <ghisvail@gmail.com>
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions
#are met:
#1. Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#3. Neither the name of the linop developers nor the names of any contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
#OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
#SUCH DAMAGE.

from __future__ import division
from numpy.testing import TestCase, assert_, assert_equal, assert_raises
import numpy as np
import pykrylov.linop as lo
from pykrylov.linop import ShapeError
from pykrylov.tools.types import allowed_types


def get_matvecs(A):
    return {'shape': A.shape,
            'matvec': lambda x: np.dot(A, x),
            'matvec_transp': lambda x: np.dot(A.T, x)}


class TestLinearOperator(TestCase):
    def setUp(self):
        self.A = np.array([[1, 2, 3],
                           [4, 5, 6]])
        self.B = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
        self.C = np.array([[1, 2],
                           [3, 4]])

    def test_init(self):
        matvecs = get_matvecs(self.A)
        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                              nargout=matvecs['shape'][0],
                              matvec=matvecs['matvec'])
        assert_(hasattr(A, '_matvec'))
        assert_(hasattr(A, 'dtype'))
        assert_(A.T is None)
        assert_(A.H is None)

        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                              nargout=matvecs['shape'][0],
                              matvec=matvecs['matvec'],
                              matvec_transp=matvecs['matvec_transp'])
        assert_(A.T is not None)
        assert_(A.H is A.T)

        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                              nargout=matvecs['shape'][0],
                              matvec=matvecs['matvec'])
        B = lo.LinearOperator(nargin=matvecs['shape'][0],
                              nargout=matvecs['shape'][1],
                              matvec=matvecs['matvec_transp'],
                              transpose_of=A)
        assert_(B.T is A)

        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                              nargout=matvecs['shape'][0],
                              matvec=matvecs['matvec'])
        B = lo.LinearOperator(nargin=matvecs['shape'][0],
                              nargout=matvecs['shape'][1],
                              matvec=matvecs['matvec_transp'],
                              transpose_of=A)
        assert_(B.H is A)

    def test_runtime(self):
        matvecs = get_matvecs(self.A)
        A = lo.LinearOperator(nargin=matvecs['shape'][1],
                              nargout=matvecs['shape'][0],
                              matvec=matvecs['matvec'],
                              matvec_transp=matvecs['matvec_transp'])

        matvecs = get_matvecs(self.B)
        B = lo.LinearOperator(nargin=matvecs['shape'][1],
                              nargout=matvecs['shape'][0],
                              matvec=matvecs['matvec'],
                              matvec_transp=matvecs['matvec_transp'])

        matvecs = get_matvecs(self.C)
        C = lo.LinearOperator(nargin=matvecs['shape'][1],
                              nargout=matvecs['shape'][0],
                              matvec=matvecs['matvec'],
                              matvec_transp=matvecs['matvec_transp'])

        u = np.array([1, 1])
        v = np.array([1, 1, 1])
        assert_equal(A * v, [6, 15])
        assert_equal(A._matvec(v), [6, 15])
        assert_equal(A.T * u, [5, 7, 9])
        assert_equal(A.H * u, [5, 7, 9])
        assert_equal((A * 2) * v, A * (2 * v))
        assert_equal((A * 2) * v, (2 * A) * v)
        assert_equal((A / 2) * v, A * (v / 2))
        assert_equal((-A) * v, A * (-v))
        assert_equal((A - A) * v, [0, 0])
        assert_equal((C ** 2) * u, [17, 37])
        assert_equal((C ** 2) * u, (C * C) * u)

        assert_(isinstance(A + A, lo.LinearOperator))
        assert_(isinstance(A - A, lo.LinearOperator))
        assert_(isinstance(-A, lo.LinearOperator))
        assert_(isinstance(2 * A, lo.LinearOperator))
        assert_(isinstance(A * 2, lo.LinearOperator))
        assert_(isinstance(A * 0, lo.ZeroOperator))
        assert_(isinstance(A / 2, lo.LinearOperator))
        assert_(isinstance(C ** 2, lo.LinearOperator))
        assert_(isinstance(C ** 0, lo.IdentityOperator))

        sum_A = lambda x: A + x
        assert_raises(ValueError, sum_A, 3)
        assert_raises(ValueError, sum_A, v)
        assert_raises(ShapeError, sum_A, B)

        sub_A = lambda x: A - x
        assert_raises(ValueError, sub_A, 3)
        assert_raises(ValueError, sub_A, v)
        assert_raises(ShapeError, sub_A, B)

        mul_A = lambda x: A * x
        assert_raises(ValueError, mul_A, u)
        assert_raises(ShapeError, mul_A, A)

        div_A = lambda x: A / x
        assert_raises(ValueError, div_A, B)
        assert_raises(ValueError, div_A, u)
        assert_raises(ZeroDivisionError, div_A, 0)

        pow_A = lambda x: A ** x
        pow_C = lambda x: C ** x
        assert_raises(ShapeError, pow_A, 2)
        assert_raises(ValueError, pow_C, -2)
        assert_raises(ValueError, pow_C, 2.1)

    def test_dtypes(self):
        for dtype_op in allowed_types:
            for dtype_in in allowed_types:
                dtype_out = np.result_type(dtype_op, dtype_in)

                matvecs = get_matvecs(self.A)
                A = lo.LinearOperator(nargin=matvecs['shape'][1],
                                      nargout=matvecs['shape'][0],
                                      matvec=matvecs['matvec'],
                                      matvec_transp=matvecs['matvec_transp'],
                                      dtype=dtype_op)
                x = np.array([1, 1, 1]).astype(dtype_in)
                assert_((A * x).dtype == dtype_out)


class TestIdentityOperator(TestCase):
    def test_runtime(self):
        A = lo.IdentityOperator(3)
        x = np.array([1, 1, 1])
        assert_equal(A * x, x)
        assert_(A.T is A)
        assert_(A.H is A)

    def test_dtypes(self):
        for dtype_op in allowed_types:
            for dtype_in in allowed_types:
                dtype_out = np.result_type(dtype_op, dtype_in)
                A = lo.IdentityOperator(3, dtype=dtype_op)
                x = np.array([1, 1, 1]).astype(dtype_in)
                assert_((A * x).dtype == dtype_out)


class TestDiagonalOperator(TestCase):
    def test_init(self):
        A_diag = [1, 2, 3]
        A = lo.DiagonalOperator(A_diag)
        assert_(A.shape == (len(A_diag), len(A_diag)))
        self.assertTrue(A.symmetric)
        self.assertTrue(A.hermitian)

    def test_runtime(self):
        A = lo.DiagonalOperator([1, 2, 3])
        x = np.array([1, 1, 1])
        assert_equal(A * x, [1, 2, 3])
        assert_equal(A.H * x, [1, 2, 3])
        assert_(A.T is A)
        assert_(A.H is A)
        assert_raises(ValueError, lo.DiagonalOperator, 10)
        assert_raises(ValueError, lo.DiagonalOperator, np.eye(3))

    def test_dtypes(self):
        for dtype_op in allowed_types:
            for dtype_in in allowed_types:
                dtype_out = np.result_type(dtype_op, dtype_in)
                diag = np.array([1, 2, 3]).astype(dtype_op)
                A = lo.DiagonalOperator(diag)
                x = np.array([1, 1, 1]).astype(dtype_in)
                assert_((A * x).dtype == dtype_out)


class TestZeroOperator(TestCase):
    def test_runtime(self):
        A = lo.ZeroOperator(2, 3)
        x = np.array([1, 1])
        assert_equal(A * x, [0, 0, 0])
        x = np.array([1, 1, 1])
        assert_equal(A.T * x, [0, 0])
        assert_equal(A.H * x, [0, 0])

    def test_dtypes(self):
        for dtype_op in allowed_types:
            for dtype_in in allowed_types:
                dtype_out = np.result_type(dtype_op, dtype_in)
                A = lo.ZeroOperator(3, 3, dtype=dtype_op)
                x = np.array([1, 1, 1]).astype(dtype_in)
                assert_((A * x).dtype == dtype_out)


class TestReducedLinearOperator(TestCase):
    def test_init(self):
        v = np.arange(6)
        A = lo.linop_from_ndarray(np.outer(v, v))
        R = lo.ReducedLinearOperator(A, [1, 2, 3], [4, 5])
        self.assertFalse(R.symmetric)
        self.assert_(R.shape == (3, 2))

    def test_runtime(self):
        v = np.arange(6)
        A_mat = np.outer(v, v)
        A = lo.linop_from_ndarray(A_mat)
        row_idx = [2, 4]
        col_idx = [1, 2, 3]
        R = lo.ReducedLinearOperator(A, row_idx, col_idx)
        vmask = np.zeros(v.size); vmask[col_idx] = 1
        assert_equal(R * v[col_idx], A_mat.dot(v * vmask)[row_idx])


class TestSymmetricallyReducedLinearOperator(TestCase):
    def test_init(self):
        v = np.arange(6)
        A = lo.linop_from_ndarray(np.outer(v, v), symmetric=True)
        R = lo.SymmetricallyReducedLinearOperator(A, [1, 2, 3])
        self.assertTrue(R.symmetric)
        self.assertTrue(R.hermitian)
        A = lo.linop_from_ndarray(np.outer(v, v))
        R = lo.SymmetricallyReducedLinearOperator(A, [1, 2, 3])
        self.assertFalse(R.symmetric)
        self.assert_(R.shape == (3, 3))

    def test_runtime(self):
        v = np.arange(6)
        A_mat = np.outer(v, v)
        A = lo.linop_from_ndarray(A_mat)
        sym_idx = [1, 2, 3]
        R = lo.SymmetricallyReducedLinearOperator(A, sym_idx)
        vmask = np.zeros(v.size); vmask[sym_idx] = 1
        assert_equal(R * v[sym_idx], A_mat.dot(v * vmask)[sym_idx])


def test_linop_from_ndarray():
    A = np.array([[1, 2, 3],
                 [4, 5, 6]])
    A_as_op = lo.linop_from_ndarray(A)
    assert_(isinstance(A_as_op, lo.LinearOperator))
    x = np.array([1, 1, 1])
    assert_equal(A_as_op * x, A.dot(x))
    x = np.array([1, 1])
    assert_equal(A_as_op.T * x, A.T.dot(x))
    assert_equal(A_as_op.H * x, A.T.dot(x))

    init_Aop = lambda s, h: lo.linop_from_ndarray(A, symmetric=s, hermitian=h)
    assert_raises(ValueError, init_Aop, True, False)
