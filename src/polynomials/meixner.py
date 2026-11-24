
from classes.ortho_polynomials_class import OrthogonalPolynomials
from scipy import special
from mpmath import *

class MeixnerPolynomials(OrthogonalPolynomials):
    def __init__(self, g, m):
        OrthogonalPolynomials.__init__(self, g, m)
        self.g = g
        self.m = m

    def beta_n(self, n):
        return (
            (n * (1 + self.m) + self.m * self.g) /
            (1 - self.m)
        )

    def gamma_n(self, n):
       return (
           n * self.m *
           (n - 1 + self.g) /
           (self.m - 1) ** 2
       )

    def squared_norm(self, n):
        if n == -1:
            return 1
        else:
            return (
                factorial(n) *
                special.poch(self.g, n) *
                self.m ** n /
                (1 - self.m) ** (self.g + 2 * n)
            )

    def weight_function(self, x):
        return (
            self.m ** x * gamma(self.g + x) /
            (
                gamma(self.g) * gamma(x + 1)
            )
        )

    def d_operator_monic_polynomials(self, x, n, k):
        meix_inst = MeixnerPolynomials(self.g + 1, self.m)
        return (
            ff(n, k) *
            meix_inst.eval_monic_polynomials(x, n - k)
        )