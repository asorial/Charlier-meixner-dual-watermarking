
from classes.sobolev_type_polynomials_one_mass import SobolevTypePolynomialsIMass
from polynomials.meixner import MeixnerPolynomials
from mpmath import *

mp.dps = 25; mp.pretty = True

class MeixnerSobolevTypePolynomialsIMass(SobolevTypePolynomialsIMass, MeixnerPolynomials):
    def __init__(self, lamb, alpha, j, g, m):
        SobolevTypePolynomialsIMass.__init__(self, lamb, alpha, j, g, m)
        MeixnerPolynomials.__init__(self, g, m)   