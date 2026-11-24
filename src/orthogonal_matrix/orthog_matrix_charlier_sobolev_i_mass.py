
from orthogonal_matrix.orthog_matrix import OrthogonalMatrix
from sobolev_type_polynomials.charlier_sobolev_type_polynomials_one_mass import CharlierSobolevTypePolynomialsIMass
from mpmath import *

mp.dps = 25; mp.pretty = True

class OrthogonalMatrixCharlierSobolevTypePolynomialsIMass(OrthogonalMatrix, CharlierSobolevTypePolynomialsIMass):
    def __init__(self, sblock, lamb, alpha, j, a):
        OrthogonalMatrix.__init__(self, sblock,  lamb, alpha, j, a)
        CharlierSobolevTypePolynomialsIMass.__init__(self,  lamb, alpha, j, a)

    def polynomial_func(self, x, n):
        return self.eval_monic_sobolev_type_polynomials(x, n)