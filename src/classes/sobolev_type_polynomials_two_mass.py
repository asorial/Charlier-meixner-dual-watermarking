
from classes.to_obtain_connection_formula_two_mass import ToObtainConnetionFormulaIIM
from classes.sobolev_type_polynomials import SobolevTypePolynomials
from mpmath import *

mp.dps = 25; mp.pretty = True

class SobolevTypePolynomialsIIMass(ToObtainConnetionFormulaIIM, SobolevTypePolynomials):
    def __init__(self, mu, lamb, alpha, beta, i, j, *args):
        ToObtainConnetionFormulaIIM.__init__(self, mu, lamb, alpha, beta, i, j, *args)

    def factorial_func(self, n):
        return fac(n)
    
    def qsubtraction(self, x, y, n):
        return ff(x - y, n)