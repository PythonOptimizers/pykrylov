from math import sqrt

def Poisson1dMatvec(x):
    # Matrix-vector product with a 1D Poisson matrix
    y = 2*x
    y[:-1] -= x[1:]
    y[1:] -= x[:-1]
    return y

def Poisson2dMatvec(x):
    # Matrix-vector product with a 2D Poisson matrix
    n = int(sqrt(x.shape[0]))
    # Contribution of main diagonal and off-diagonal blocks
    y = 4*x
    y[n:] -= x[:-n]
    y[:-n] -= x[n:]
    # Contribution of first diagonal block
    y[:n-1] -= x[1:n]
    y[1:n] -= x[:n-1]
    # Contribution of intermediate diagonal blocks
    for i in xrange(1,n-1):
        xi = x[i*n:(i+1)*n]   # This a view of x, not a copy
        yi = y[i*n:(i+1)*n]
        yi[:-1] -= xi[1:]
        yi[1:] -= xi[:-1]
    # Contribution of last diagonal block
    y[(n-1)*n+1:] -= x[(n-1)*n:-1]
    y[(n-1)*n:n*n-1] -= x[-n+1:]
    return y
