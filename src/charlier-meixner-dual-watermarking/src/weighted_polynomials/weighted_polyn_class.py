
from classes.ortho_polynomials_class import OrthogonalPolynomials
from math import sqrt
from mpmath import *

mp.dps = 25; mp.pretty = True

class WeightedPolynomials(OrthogonalPolynomials):
    def __init__(self, *args):
        OrthogonalPolynomials.__init__(self, *args)

    def polynomial_func(self, x, n):
        pass

    def weighted_polyn(self, x, n):
        return (
            self.polynomial_func(x, n) *
            sqrt(
                self.weight_function(x) /
                self.squared_norm(n)
            )
        )