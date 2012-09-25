.. Description of linear operators
.. _linop-page:

================
Linear Operators
================

Introduction
============

When working towards a solution of a linear system :math:`Ax=b`, Krylov methods
do not need to know anything structural about the matrix :math:`A`; all they
require is the ability to form matrix-vector products :math:`v \mapsto Av` and,
possibly, products with the transpose :math:`u \mapsto A^T u`. In essence, we
do not even need the *operator* :math:`A` to be represented by a matrix at all;
we simply consider it as a linear function.

In PyKrylov, such linear functions can be conveniently packaged as
``LinearOperator`` objects. If ``A`` is an instance of ``LinearOperator`` and
represents the "matrix" :math:`A` above, we may computes matrix-vector products
by simply writing ``A*v``, where ``v`` is a Numpy array of appropriate size.

Similarly, if a Krylov method requires access to the transpose operator
:math:`A^T`, it is conveniently available as ``A.T`` and products may be
computed using, e.g., ``A.T * u``. If ``A`` represents a symmetric operator
:math:`A = A^T`, then ``A.T`` is simply a reference to ``A`` itself.

More generally, since :math:`(A^T)^T = A`, the Python statement ``A.T.T is A``
always evaluates to ``True``, which means that they are the *same* object.

In the next two sections, we describe generic linear operators and linear
operators constructed by blocks.

The :mod:`linop` Module
=======================

.. automodule:: linop

Base Class for Linear Operators
-------------------------------

All linear operators derive from the base class ``BaseLinearOperator``. This
base class is not meant to be used directly to define linear operators, other
than by subclassing to define classes of more specific linear operators.

.. autoclass:: BaseLinearOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

Linear Operators Defined by Functions
-------------------------------------

It is intuitive to define an operator by its *action* on vectors. The
``LinearOperator`` class takes arguments ``matvec`` and ``matvec_transp`` to
define the action of the operator and of its transpose.

Here is a simple example:

.. code-block:: python

    import numpy as np
    A = LinearOperator(nargin=3, nargout=3, matvec=lambda v: 2*v, symmetric=True)
    B = LinearOperator(nargin=4, nargout=3, matvec=lambda v: np.arange(3)*v[:3],
                       matvec_transp=lambda v: np.concatenate((np.arange(3)*v, np.zeros(1))))

Here, ``A`` represents the operator :math:`2I`, where :math:`I` is the identity
and ``B`` could be represented by the matrix

.. math::

    \begin{bmatrix}
      1 & & & \\
        & 2 & & \\
        & & 3 & 0 \\
    \end{bmatrix}.

Note that any callable object can be used to pass values for ``matvec`` and
``matvec_transp``. For example :

.. code-block:: python

    def func(v):
        return np.arange(3) * v

    class MyClass(object):
        def __call__(self, u):
            return np.concatenate((np.arange(3)*v, np.zeros(1)))

    myobject = MyClass()
    B = LinearOperator(nargin=4, nargout=3, matvec=func, matvec_transp=myobject)


is perfectly valid. Based on this example, arbitrarily complex operators may be
built.

.. autoclass:: LinearOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

Simple Common Predefined Linear Operators
-----------------------------------------

A few common operators are predefined, such as the identity, the zero operator,
and a class for diagonal operators.

.. autoclass:: IdentityOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

.. autoclass:: ZeroOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

Diagonal operators are simply defined by their diagonal as a Numpy array. For
example:

.. code-block:: python

    d = np.random.random(10)
    D = DiagonalOperator(d)

.. autoclass:: DiagonalOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

Convenience Functions
---------------------

Typically, linear operators don't come alone and an operator is often used to
define other operators. An example is reduction. Suppose :math:`A` is a linear
operator from :math:`\mathbb{R}^n` into :math:`\mathbb{R^m}`,
:math:`\mathcal{Z}` is a subspace of :math:`\mathbb{R}^n` and
:math:`\mathcal{Y}` is a subspace of :math:`\mathbb{R}^m`. Sometimes it is
useful to consider :math:`A` restricted to :math:`\mathcal{Z}` and
co-restricted to :math:`\mathcal{Y}`. Assuming that :math:`A` is a matrix
representing the linear operator and :math:`Z` and :math:`Y` are matrices whose
columns form bases of the subspaces :math:`\mathcal{Z}` and
:math:`\mathcal{Y}`, respectively, then the restricted operator may be written
:math:`Y^T A Z`.

A simple version of this type of reduction is where we only consider a subset
of the rows and columns of the matrix :math:`A`, which corresponds to subspaces
:math:`\mathcal{Z}` and :math:`\mathcal{Y}` aligned with the axes of
coordinates.

Note that by default, the reduced linear operator is considered to be
non-symmetric even if the original operator was symmetric.

.. autofunction:: ReducedLinearOperator

A special case of this type of reduction is when ``row_indices`` and
``col_indices`` are the same. This is often useful in combination with square
symmetric operators. In this case, the reduced operator possesses the same
symmetry as the original operator.

.. autofunction:: SymmetricallyReducedLinearOperator

An obvious use case of linear operators is matrices themselves! The two
following convenience functions build linear operators from `Pysparse
<http://pysparse.sf.net>`_ sparse matrices and from Numpy arrays.

.. autofunction:: PysparseLinearOperator

.. autofunction:: linop_from_ndarray

Note that there is normally no need to build linear operators from Numpy
matrices or from Scipy sparse matrices since they already support product and
transposition.

Exceptions
----------

.. autoexception:: ShapeError


Block Linear Operators: The :mod:`blkop` Module
===============================================

Linear operators are sometimes defined by blocks. This is often the case in
numerical optimization and the solution of partial-differential equations. An
example of operator defined by blocks is

.. math::

    K =
    \begin{bmatrix}
      A & B \\
      C & D
    \end{bmatrix}

where :math:`A`, :math:`B`, :math:`C` and :math:`D` are linear operators
(perhaps themselves defined by blocks) of appropriate shape.

The general class ``BlockLinearOperator`` may be used to represent the operator
above. If more structure is present, for example if the off-diagonal blocks are
zero, :math:`K` is a block-diagonal operator and the class
``BlockDiagonalLinearOperator`` may be used to define it.

.. automodule:: blkop

General Block Operators
-----------------------

General block operators are defined using a list of lists, each of which
defines a block row. If the block operator is specified as symmetric, each
block on the diagonal must be symmetric. For example:

.. code-block:: python

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

    # Build [A  B].
    K1 = BlockLinearOperator([[A, B]])

    # Build [A  B]
    #       [C  D].
    K2 = BlockLinearOperator([[A, B], [C, D]])

    # Build [A]
    #       [C].
    K3 = BlockLinearOperator([[A], [C]])

    # Build [A  B]
    #       [B' E].
    K4 = BlockLinearOperator([[A, B], [E]], symmetric=True)


.. autoclass:: BlockLinearOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:

Block Diagonal Operators
------------------------

Block diagonal operators are a special case of block operators and are defined
with a list containing the blocks on the diagonal. If the block operator is
specified as symmetric, each block must be symmetric. For example:

.. code-block:: python

    K5 = BlockDiagonalLinearOperator([A, E], symmetric=True)

.. autoclass:: BlockDiagonalLinearOperator
   :show-inheritance:
   :members:
   :inherited-members:
   :undoc-members:


Operations with Operators
=========================

Linear operators, whether defined by blocks or not, may be added together or
composed following the usual rules of linear algebra. An operator may be
multiplied by a scalar or by another operator. Operators of the same shape may
be added or subtracted. Those operations are essentially free in the sense that
a new linear operator results of them, which encapsulates the appropriate rules
for multiplication by a vector. It is only when the resulting operator is
applied to a vector that the appropriate chain of operations is applied. For
example:

.. code-block:: python

    AB = A * B
    AA = A * A.T
    G  = E + 2 * B.T * B

Block operators also support iteration and indexing. Iterating over a block
operator amounts to iterating row-wise over its blocks. Iterating over a block
diagonal operator amounts to iterating over its diagonal blocks. Indexing works
as expected. Indexing general block operators requires two indices, much as
when indexing a matrix, while indexing a block diagonal operator requires a
single indices. For example:

.. code-block:: python

    K2 = BlockLinearOperator([[A, B], [C, D]])
    K2[0,:]   # Returns the block operator defined by [[A, B]].
    K2[:,1]   # Returns the block operator defined by [[C], [D]].
    K2[1,1]   # Returns the linear operator D.

    K4 = BlockLinearOperator([[A, B], [E]], symmetric=True)
    K4[0,1]   # Returns the linear operator B.T.

    K5 = BlockDiagonalLinearOperator([A, E], symmetric=True)
    K5[0]     # Returns the linear operator A.
    K5[1]     # Returns the linear operator B.
    K5[:]     # Returns the diagonal operator defines by [A, E].
