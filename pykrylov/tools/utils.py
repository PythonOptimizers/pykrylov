# Various utilities.

import numpy
from math import copysign, sqrt

def roots_quadratic(q2, q1, q0, tol=1.0e-8, nitref=1):
    """
    Find the real roots of the quadratic q(x) = q2 * x^2 + q1 * x + q0.
    The numbers a0, a1 and a0 must be real.

    This function is written after the GALAHAD function of the same name.
    See http://galahad.rl.ac.uk.
    """
    a2 = float(q2) ; a1 = float(q1) ; a0 = float(q0)

    # Case of a linear function.
    if a2 == 0.0:
        if a1 == 0.0:
            if a0 == 0.0:
                return [0.0]
            else:
                return []
        else:
            roots = [-a0/a1]
    else:
        # Case of a quadratic.
        rhs = tol * a1 * a1
        if abs(a0*a2) > rhs:
            rho = a1 * a1 - 4.0 * a2 * a0
            if rho < 0.0:
                return []
            # There are two real roots.
            d = -0.5 * (a1 + copysign(sqrt(rho), a1))
            roots = [d/a2, a0/d]
        else:
            # Ill-conditioned quadratic.
            roots = [-a1/a2, 0.0]

    # Perform a few Newton iterations to improve accuracy.
    new_roots = []
    for root in roots:
        for it in range(nitref):
            val = (a2 * root + a1) * root + a0
            der = 2.0 * a2 * root + a1
            if der == 0.0:
                continue
            else:
                root = root - val/der
        new_roots.append(root)

    return new_roots


def check_symmetric(op, x=None):
    """
    Cheap check that a linear operator is symmetric. Supply `op`, a callable
    linear operator and `x`, an initial vector. If `x` is not supplied, a
    random vector will be generated. This function returns `True` or `False`.
    """
    m, n = op.get_shape()
    if m != n: return False
    eps = numpy.finfo(numpy.double).eps
    if x is None:
        x = numpy.random.random(n)
    w = op(x)
    r = op(w)
    s = numpy.dot(w,w)
    t = numpy.dot(x,r)
    z = abs(s - t)
    epsa = (s + eps) * eps**(1.0/3)
    return (z <= epsa)



if __name__ == '__main__':
    roots = roots_quadratic(2.0e+20,.1,-4)
    print 'Received: ', roots
