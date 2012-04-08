# Demo for CG.
# Requires Pysparse (http://pysparse.sf.net)
# The test matrix may be obtained from http://math.nist.gov/MatrixMarket

from pykrylov.cg import CG
from demo_common import demo
import logging
import sys

if __name__ == '__main__':

    # Create logger for CG.
    cglog = logging.getLogger('CG')
    cglog.setLevel(logging.INFO)
    fmt = logging.Formatter('%(name)-2s %(levelname)-8s %(message)s')
    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)
    cglog.addHandler(hndlr)

    demo(CG, sys.argv[1], check_symmetric=True, logger=cglog)
