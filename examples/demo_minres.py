# Demo for MINRES.
# Requires Pysparse (http://pysparse.sf.net)
# The test matrix may be obtained from http://math.nist.gov/MatrixMarket

from pykrylov.minres import Minres
from demo_common import demo
import logging
import sys

if __name__ == '__main__':

    # Create logger for MINRES.
    mrlog = logging.getLogger('MINRES')
    mrlog.setLevel(logging.INFO)
    fmt = logging.Formatter('%(name)-2s %(levelname)-8s %(message)s')
    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)
    mrlog.addHandler(hndlr)

    demo(Minres, sys.argv[1], check_symmetric=True, logger=mrlog)
