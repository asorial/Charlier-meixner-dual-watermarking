
from classes.sobolev_type_polynomials_one_mass import SobolevTypePolynomialsIMass
from polynomials.charlier import CharlierPolynomials
from mpmath import *

mp.dps = 25; mp.pretty = True

class CharlierSobolevTypePolynomialsIMass(SobolevTypePolynomialsIMass, CharlierPolynomials):
    def __init__(self, lamb, alpha, j, a):
        SobolevTypePolynomialsIMass.__init__(self, lamb, alpha, j, a)
        CharlierPolynomials.__init__(self, a)   