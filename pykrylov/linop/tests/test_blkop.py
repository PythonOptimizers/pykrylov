"""Test suite for the blkop module."""
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
import unittest
from numpy.testing import assert_, assert_equal, assert_raises
import numpy as np
import pykrylov.linop as lo
from pykrylov.linop import blkop as bo
from pykrylov.linop import ShapeError


class TestBlockLinearOperator(unittest.TestCase):
    def setUp(self):
        self.A = lo.IdentityOperator(2)
        self.B = lo.linop_from_ndarray(np.arange(1, 7).reshape([2, 3]))
        self.C = lo.DiagonalOperator(np.arange(3))
        self.D = lo.linop_from_ndarray(np.arange(6, 0, -1).reshape([3, 2]))

    def test_init(self):
        M = bo.BlockLinearOperator([[self.A, self.B], [self.D, self.C]])
        assert_(M.shape == (5, 5))
        assert_(self.A in M)
        assert_(M[0, 0] is self.A)
        assert_(self.B in M)
        assert_(M[0, 1] is self.B)
        assert_(self.C in M)
        assert_(M[1, 1] is self.C)
        assert_(self.D in M)
        assert_(M[1, 0] is self.D)

        M = bo.BlockLinearOperator([[self.A, self.B], [self.C]], symmetric=True)
        assert_(M.shape == (5, 5))
        assert_(self.B.T in M)
        assert_(M[1, 0] is self.B.T)

        M = bo.BlockLinearOperator([[self.A, self.B], [self.A, self.B]])
        assert_(M.shape == (4, 5))

        assert_raises(TypeError, bo.BlockLinearOperator,
                      [self.A, self.C, self.D, self.C])
        assert_raises(ShapeError, bo.BlockLinearOperator,
                      [[self.A, self.C], [self.D, self.C]])
        assert_raises(ShapeError, bo.BlockLinearOperator,
                      [[self.A, self.B], [self.B, self.C]])
        assert_raises(ValueError, bo.BlockLinearOperator,
                      [[self.A, self.B], [self.B]], symmetric=True)

    def test_runtime(self):
        M = bo.BlockLinearOperator([[self.A, self.B], [self.D, self.C]])
        matrix_M = np.array([[1, 0, 1, 2, 3],
                             [0, 1, 4, 5, 6],
                             [6, 5, 0, 0, 0],
                             [4, 3, 0, 1, 0],
                             [2, 1, 0, 0, 2]])
        x = np.random.random(M.shape[1])
        assert_(np.allclose(M * x, np.dot(matrix_M, x)))
        x = np.random.random(M.T.shape[1])
        assert_(np.allclose(M.T * x, np.dot(matrix_M.T, x)))
        assert_(np.allclose(M.H * x, np.dot(matrix_M.T, x)))

        M = bo.BlockLinearOperator([[self.A, self.B], [self.C]], symmetric=True)
        matrix_M = np.array([[1, 0, 1, 2, 3],
                             [0, 1, 4, 5, 6],
                             [1, 4, 0, 0, 0],
                             [2, 5, 0, 1, 0],
                             [3, 6, 0, 0, 2]])
        x = np.random.random(M.shape[1])
        assert_(np.allclose(M * x, np.dot(matrix_M, x)))
        x = np.random.random(M.T.shape[1])
        assert_(np.allclose(M.T * x, np.dot(matrix_M.T, x)))
        assert_(np.allclose(M.H * x, np.dot(matrix_M.T, x)))

        M = bo.BlockLinearOperator([[self.A, self.B], [self.A, self.B]])
        matrix_M = np.array([[1, 0, 1, 2, 3],
                             [0, 1, 4, 5, 6],
                             [1, 0, 1, 2, 3],
                             [0, 1, 4, 5, 6]])
        x = np.random.random(M.shape[1])
        assert_(np.allclose(M * x, np.dot(matrix_M, x)))
        x = np.random.random(M.T.shape[1])
        assert_(np.allclose(M.T * x, np.dot(matrix_M.T, x)))
        assert_(np.allclose(M.H * x, np.dot(matrix_M.T, x)))

    def test_dtypes(self):
        pass


class TestBlockDiagonalOperator(unittest.TestCase):
    def setUp(self):
        self.A = lo.IdentityOperator(2)
        self.B = lo.linop_from_ndarray(np.arange(1, 7).reshape([2, 3]))
        self.C = lo.DiagonalOperator(np.arange(3))
        self.D = lo.linop_from_ndarray(np.arange(6, 0, -1).reshape([3, 2]))

    def test_init(self):
        M = bo.BlockDiagonalLinearOperator([self.A, self.C])
        assert_(M.shape == (5, 5))
        assert_(M.symmetric is True)
        assert_(self.A in M)
        assert_(M[0] is self.A)
        assert_(self.C in M)
        assert_(M[1] is self.C)

        M = bo.BlockDiagonalLinearOperator([self.A, self.D*self.B])
        assert_(M.symmetric is False)
        assert_(M.hermitian is False)

        assert_raises(AttributeError, bo.BlockDiagonalLinearOperator,
                      [[self.A, self.C]])

    def test_runtime(self):
        M = bo.BlockDiagonalLinearOperator([self.A, self.C])
        matrix_M = np.array([[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 2]])
        x = np.ones(M.shape[1])
        assert_equal(M * x, np.dot(matrix_M, x))
        x = np.ones(M.T.shape[1])
        assert_equal(M.T * x, np.dot(matrix_M.T, x))


if __name__ == '__main__':
    unittest.main()
