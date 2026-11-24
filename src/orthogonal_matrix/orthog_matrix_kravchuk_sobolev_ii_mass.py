
from orthogonal_matrix.orthog_matrix import OrthogonalMatrix
from sobolev_type_polynomials.kravchuk_sobolev_type_polynomials_two_mass import KravchukSobolevTypePolynomialsIIMass
from mpmath import *

mp.dps = 25; mp.pretty = True

class OrthogonalMatrixKravchukSobolevTypePolynomialsIIMass(OrthogonalMatrix, KravchukSobolevTypePolynomialsIIMass):
    def __init__(self, sblock, mu, lamb, alpha, beta, i, j, p, N):
        OrthogonalMatrix.__init__(self, sblock, mu, lamb, alpha, beta, i, j, p, N)
        KravchukSobolevTypePolynomialsIIMass.__init__(self, mu, lamb, alpha, beta, i, j, p, N)

    def polynomial_func(self, x, n):
        return self.eval_monic_sobolev_type_polynomials(x, n)