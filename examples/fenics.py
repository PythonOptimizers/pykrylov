"""Iterative solution of a discretized PDE.

An example using a linear system arising from the
discretization of a partial differential equation.
The PDE is modeled and discretized with FEniCS.
See http://www.fenics-project.org
"""

from dolfin import *
import numpy as np

# Import a Krylov subspace solver.
# from pykrylov.cg import CG as KSolver
from pykrylov.bicgstab import BiCGSTAB as KSolver
# from pykrylov.tfqmr import TFQMR as KSolver
# from pykrylov.cgs import CGS as KSolver
from pykrylov.linop import LinearOperator, DiagonalOperator


class LeftRightBoundary(SubDomain):
    """Define Dirichlet boundary conditions."""

    def inside(self, x, on_boundary):
        """Boundary conditions at x = 0 or x = 1."""
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS


def setup_problem(nx, ny, degree=1):
    """Setup variational problem."""
    # Define domain.
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, "Lagrange", degree=degree)

    # Define BCs.
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, LeftRightBoundary())

    # Define variational problem.
    v = TestFunction(V)
    u = TrialFunction(V)  # "trial function" = "unknown".
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
    g = Expression("sin(5 * x[0])")
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx + g * v * ds
    return (mesh, a, L, bc, u0, V)


def setup_system(a, L, bc):
    """Assemble variational problem and apply boundary conditions."""
    A = assemble(a)
    b = assemble(L)
    bc.apply(A, b)
    return (A, b)


if __name__ == '__main__':

    # Assemble linear system.
    (mesh, a, L, bc, u0, V) = setup_problem(32, 32, degree=1)
    (A, b) = setup_system(a, L, bc)

    # Define operator and diagonal preconditioner.
    Aop = LinearOperator(A.size(0), A.size(1),
                         lambda v: A * v,
                         matvec_transp=lambda u: A.transpmult(u),
                         symmetric=False)
    D = DiagonalOperator(1 / np.diag(A.array()))

    # Solve with preconditioned CG.
    ksolver = KSolver(Aop, precon=D)
    ksolver.solve(b.array())

    print 'Using ' + ksolver.name
    print 'System size: ', Aop.shape[0]
    print 'Converged: ', ksolver.converged
    print 'Initial/Final Residual:  %7.1e/%7.1e' \
        % (ksolver.residNorm0, ksolver.residNorm)
    print 'Matvecs:   ', ksolver.nMatvec

    # Plot solution
    u = Function(V)
    u.vector()[:] = ksolver.bestSolution
    plot(u, interactive=True)
