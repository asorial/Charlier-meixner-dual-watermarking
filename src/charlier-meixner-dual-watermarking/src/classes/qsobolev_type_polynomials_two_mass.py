
from classes.to_obtain_connection_formula_two_mass import ToObtainConnetionFormulaIIM
from classes.sobolev_type_polynomials import SobolevTypePolynomials
from mpmath import *
import math

mp.dps = 25; mp.pretty = True

class SobolevTypeQPolynomialsIIMass(ToObtainConnetionFormulaIIM, SobolevTypePolynomials):
    def __init__(self, mu, lamb, alpha, beta, i, j, *args):
        ToObtainConnetionFormulaIIM.__init__(self, mu, lamb, alpha, beta, i, j, *args)
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