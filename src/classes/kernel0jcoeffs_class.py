
from classes.ortho_polynomials_class import OrthogonalPolynomials
from mpmath import *

mp.dps = 25; mp.pretty = True

class Kernel0JCoefficients(OrthogonalPolynomials):
    def __init__(self, *args):
        OrthogonalPolynomials.__init__(self, *args)

    def factorial_func(self, n):
        pass
    
    def qsubtraction(self, x, y, n):
        pass

    def An(self, x, y, n, l):
        return (
            self.factorial_func(l) /
            (
                self.squared_norm(n - 1) *
                self.qsubtraction(x, y, l + 1)
            ) *
            sum(
                [
                    self.d_operator_monic_polynomials(y, n - 1, k) *
                    self.qsubtraction(x, y, k) /
                    self.factorial_func(k)
                    for k in range(l + 1)
                ]
            )
        )

    def Bn(self, x, y, n, l):
        return (
            - self.factorial_func(l) /
            (
                self.squared_norm(n - 1) *
                self.qsubtraction(x, y, l + 1)
            ) *
            sum(
                [
                    self.d_operator_monic_polynomials(y, n, k) *
                    self.qsubtraction(x, y, k) /
                    self.factorial_func(k)
                    for k in range(l + 1)
                ]
            )
        )