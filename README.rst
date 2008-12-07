========
PyKrylov
========

PyKrylov is a Python package implementing common Krylov methods in pure Python.


Requirements
============

1. `Python <http://www.python.org>`_ 2.3, 2.4 or 2.5. We are not sure about
   Python 2.6 compatibility and are pretty sure about incompatibility with
   Python 3.0.
2. `NumPy <http://www.scipy.org/NumPy>`_

If you are working under Linux, OS/X or Windows, prebuilt packages are
available. Remember that for efficiency, it is recommended to compile Numpy
against optimized LAPACK and BLAS libraries.


Krylov Methods
==============

Krylov methods are iterative methods for solving (potentially large)
systems of linear equations

        A x = b

where A is a matrix and x and b are vectors of compatible dimension. Different
Krylov methods are used depending on the properties of the matrix A. Typically,
only matrix-vector products with A are required at each iteration. Some methods
require matrix-vector products with the transpose of A when the latter is not
symmetric. For more information on Krylov methods, see for instance [Kelley]_.

PyKrylov does not rely on any particular dense or sparse matrix package because 
all matrix-vector products are handled as operators, i.e., the user supplies
a function to perform such products. Similarly, preconditioners are handled as
operators and are not held explicitly.


Obtaining PyKrylov
==================



References
==========

.. [Kelley] C. T. Kelley, *Iterative Methods for Linear and Nonlinear
            Equations*, number 16 in *Frontiers in Applied Mathematics*, SIAM,
            Philadelphia, 1995.
