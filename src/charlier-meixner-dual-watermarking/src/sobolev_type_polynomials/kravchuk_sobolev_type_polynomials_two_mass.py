
from classes.sobolev_type_polynomials_two_mass import SobolevTypePolynomialsIIMass
from polynomials.kravchuk import KravchukPolynomials
from mpmath import *

mp.dps = 25; mp.pretty = True

class KravchukSobolevTypePolynomialsIIMass(SobolevTypePolynomialsIIMass, KravchukPolynomials):
    def __init__(self, mu, lamb, alpha, beta, i, j, p, N):
        SobolevTypePolynomialsIIMass.__init__(self, mu, lamb, alpha, beta, i, j, p, N)
        KravchukPolynomials.__init__(self, p, N)   