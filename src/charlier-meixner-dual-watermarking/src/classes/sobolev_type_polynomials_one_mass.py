
from classes.to_obtain_connection_formula_one_mass import ToObtainConnetionFormulaIM
from classes.sobolev_type_polynomials import SobolevTypePolynomials
from mpmath import *

mp.dps = 25; mp.pretty = True

class SobolevTypePolynomialsIMass(ToObtainConnetionFormulaIM, SobolevTypePolynomials):
    def __init__(self, lamb, alpha, j, *args):
        ToObtainConnetionFormulaIM.__init__(self, lamb, alpha, j, *args)

    def factorial_func(self, n):
        return fac(n)
    
    def qsubtraction(self, x, y, n):
        return ff(x - y, n)
    
      