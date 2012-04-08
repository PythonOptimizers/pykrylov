# Demo for CG.
# Requires Pysparse (http://pysparse.sf.net)
# The test matrix may be obtained from http://math.nist.gov/MatrixMarket

from pykrylov.cg import CG
from demo_common import demo
import sys

if __name__ == '__main__':

    demo(CG, sys.argv[1])
