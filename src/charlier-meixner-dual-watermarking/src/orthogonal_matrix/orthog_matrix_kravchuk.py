
from orthogonal_matrix.orthog_matrix import OrthogonalMatrix
from polynomials.kravchuk import KravchukPolynomials
from mpmath import *

mp.dps = 25; mp.pretty = True

class OrthogonalMatrixKravchuk(OrthogonalMatrix, KravchukPolynomials):
    def __init__(self, sblock, p, N):
        OrthogonalMatrix.__init__(self, sblock, p, N)
        KravchukPolynomials.__init__(self, p, N)

    def polynomial_func(self, x, n):
        return self.eval_monic_polynomials(x, n)