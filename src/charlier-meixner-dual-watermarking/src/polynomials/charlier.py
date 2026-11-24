
from classes.ortho_polynomials_class import OrthogonalPolynomials
from mpmath import *

mp.dps = 25; mp.pretty = True

class CharlierPolynomials(OrthogonalPolynomials):
    def __init__(self, a):
        OrthogonalPolynomials.__init__(self, a)
        self.a = a

    def beta_n(self, n):
        return n + self.a

    def gamma_n(self, n):
       return n * self.a

    def squared_norm(self, n):
        if n == -1:
            return 1
        else:
            return factorial(n) * self.a ** n

    def weight_function(self, x):
        return (
            exp(-self.a) * self.a ** x /
            gamma(x + 1)
        )

    def d_operator_monic_polynomials(self, x, n, k):
        return (
            ff(n, k) *
            self.eval_monic_polynomials(x, n - k)
        )