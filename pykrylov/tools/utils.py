# Various utilities.

from math import copysign, sqrt
#import sys

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

    #sys.stdout.write('Roots before refinement: [')
    #for root in roots:
    #    sys.stdout.write('%15.7e  ' % root)
    #sys.stdout.write(']\n')
    #sys.stdout.write('Values: [')
    #for root in roots:
    #    sys.stdout.write('%15.7e  ' % ((a2*root + a1)*root+a0))
    #sys.stdout.write(']\n')

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

    #sys.stdout.write('Roots after refinement: [')
    #for root in roots:
    #    sys.stdout.write('%15.7e  ' % root)
    #sys.stdout.write(']\n')
    #sys.stdout.write('Values: [')
    #for root in roots:
    #    sys.stdout.write('%15.7e  ' % ((a2*root + a1)*root+a0))
    #sys.stdout.write(']\n')

    return new_roots


if __name__ == '__main__':
    roots = roots_quadratic(2.0e+20,.1,-4)
    print 'Received: ', roots
