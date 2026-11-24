
from orthogonal_matrix.orthog_matrix import OrthogonalMatrix
from polynomials.meixner import MeixnerPolynomials
from mpmath import *

mp.dps = 25; mp.pretty = True

class OrthogonalMatrixMeixner(OrthogonalMatrix, MeixnerPolynomials):
    def __init__(self, sblock, g, m):
        OrthogonalMatrix.__init__(self, sblock, g, m)
        MeixnerPolynomials.__init__(self, g, m)

    def polynomial_func(self, x, n):
        return self.eval_monic_polynomials(x, n)