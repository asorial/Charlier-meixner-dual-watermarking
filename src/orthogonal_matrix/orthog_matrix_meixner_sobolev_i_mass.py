
from orthogonal_matrix.orthog_matrix import OrthogonalMatrix
from sobolev_type_polynomials.meixner_sobolev_type_polynomials_one_mass import MeixnerSobolevTypePolynomialsIMass
from mpmath import *

mp.dps = 25; mp.pretty = True

class OrthogonalMatrixMeixnerSobolevTypePolynomialsIMass(OrthogonalMatrix, MeixnerSobolevTypePolynomialsIMass):
    def __init__(self, sblock, lamb, alpha, j, g, m):
        OrthogonalMatrix.__init__(self, sblock,  lamb, alpha, j, g, m)
        MeixnerSobolevTypePolynomialsIMass.__init__(self,  lamb, alpha, j, g, m)

    def polynomial_func(self, x, n):
        return self.eval_monic_sobolev_type_polynomials(x, n)