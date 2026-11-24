
from classes.to_obtain_connection_formula_one_mass import ToObtainConnetionFormulaIM
from classes.sobolev_type_polynomials import SobolevTypePolynomials
from mpmath import *
import math

mp.dps = 25; mp.pretty = True

class SobolevTypeQPolynomialsIMass(ToObtainConnetionFormulaIM, SobolevTypePolynomials):
    def __init__(self, lamb, alpha, j, *args):
        ToObtainConnetionFormulaIM.__init__(self, lamb, alpha, j, *args)
        self.q = args[0]

    def factorial_func(self, n):
        return qfac(n, self.q)

    def qsubtraction(self, x, y, n):
        return (
            math.prod([x - y * self.q ** j for j in range(n)])
        )

    def qffactorial(self, n, k):
        return (
            qp(self.q ** -n, self.q, k) *
            self.q ** (k * n - binomial(k, 2)) /
            (self.q - 1) ** k
        )
    
      