.. PyKrylov documentation master file, created by sphinx-quickstart on Sun Dec  7 00:46:01 2008.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======================
PyKrylov Documentation
======================

:Release: |version|
:Date: |today|

.. module:: pykrylov

This is the general documentation for PyKrylov, a pure Python implementation of
common Krylov methods for the solution of systems of linear equations. The aim
of this document is to cover usage of the package and how it can serve to
benchmark solvers or preconditioners. Because the framework is very general and
does not concern any specific application, the style is light and examples keep
things simple. This is intentional so users have as much freedom as possible in
modifying the example scripts.

Some examples use the efficient `Pysparse <http://pysparse.sf.net>`_ sparse
matrix library to simulate functions that return matrix-vector products.

Contents
========

.. toctree::

   Introduction <introduction>
   Linear Operators <linop>
   Generic Template <generic>
   Conjugate Gradient <cg>
   Conjugate Gradient Squared <cgs>
   Bi-Conjugate Gradient Stabilized <bicgstab>
   Transpose-Free Quasi-Minimum Residual <tfqmr>
   Symmetric Indefinite Method with Orthogonal Factorization <symmlq>
   Benchmarking Solvers <bmark>


.. TODO List
.. =========

.. .. todolist::


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

