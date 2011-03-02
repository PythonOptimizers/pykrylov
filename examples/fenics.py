# An example using a linear system arising from the
# discretization of a partial differential equation.
# The PDE is modeled and discretized with FEniCS.
# See http://www.fenics-project.org

from dolfin import *
import numpy as np

# Import a Krylov subspace solver.
#from pykrylov.cg import CG as KSolver
#from pykrylov.bicgstab import BiCGSTAB as KSolver
#from pykrylov.tfqmr import TFQMR as KSolver
from pykrylov.cgs import CGS as KSolver

class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


def setup_problem(nx, ny, degree=1, verbose=False):
    # Define domain.
    mesh = UnitSquare(nx, ny)
    V = FunctionSpace(mesh, 'CG', degree=degree)  # degree=1: Lagrange elements.

    # Define BCs. u0 is the exact final solution.
    u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    bc = DirichletBC(V, u0, AllBoundary())

    # Define variational problem.
    v = TestFunction(V)
    u = TrialFunction(V) # "trial function" = "unknown".
    f = Expression('-8*x[0] - 10*x[1]') #Constant(-6.0)
    a = dot(grad(u), grad(v))*dx   # dx = integration over whole domain.
    L = f*v*dx

    return (mesh, a, L, bc, u0, V)


def setup_system(a, L, bc):
    A = assemble(a)
    b = assemble(L)
    bc.apply(A, b)

    return (A, b)


if __name__ == '__main__':

    # Assemble linear system.
    (mesh, a, L, bc, u0, V) = setup_problem(10, 10, degree=1, verbose=False)
    (A, b) = setup_system(a, L, bc)

    # Build diagonal preconditioner.
    diagA = np.diag(A.array())

    # Solve with preconditioned CG.
    ksolver = KSolver(lambda v: A*v, precon=lambda u: u/diagA)
    ksolver.solve(b.array())

    print 'Using ' + ksolver.name
    print 'Converged: ', ksolver.converged
    print 'Initial/Final Residual:  %7.1e/%7.1e' % (ksolver.residNorm0,ksolver.residNorm)
    print 'Matvecs:   ', ksolver.nMatvec

    # Plot solution
    u = Function(V)
    u.vector()[:] = ksolver.bestSolution
    plot(u, interactive=True)
