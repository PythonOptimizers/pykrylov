# Demo for MINRES.
# Requires Pysparse (http://pysparse.sf.net)
# The test matrix may be obtained from http://math.nist.gov/MatrixMarket

from pykrylov.minres import Minres
from demo_common import demo
import sys

if __name__ == '__main__':

    demo(Minres, sys.argv[1])
