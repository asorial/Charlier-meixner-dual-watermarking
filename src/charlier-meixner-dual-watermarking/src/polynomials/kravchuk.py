
from classes.ortho_polynomials_class import OrthogonalPolynomials
from scipy import special
from mpmath import *

mp.dps = 25; mp.pretty = True

class KravchukPolynomials(OrthogonalPolynomials):
    def __init__(self, p, N):
        OrthogonalPolynomials.__init__(self, p, N)
        self.p = p
        self.N = N

    def beta_n(self, n):
        return (
            self.N * self.p +
            (1 - 2 * self.p) * n
        )

    def gamma_n(self, n):
       return (
           n * self.p *
           (1 - self.p) *
           (self.N - n + 1)
       )

    def squared_norm(self, n):
        if n == -1:
            return 1
        else:
            return (
                factorial(n) *
                special.poch(-self.N, n) *
                self.p ** n *
                (self.p - 1) ** n
            )

    def weight_function(self, x):
        return (
            fac(self.N) * self.p ** x *
            (1 - self.p) ** (self.N - x) /
           (
               gamma(self.N - x + 1) *
               gamma(x + 1)
           )
        )

    def d_operator_monic_polynomials(self, x, n, k):
        krav_inst = KravchukPolynomials(self.p, self.N - k)
        return (
            ff(n, k) *
            krav_inst.eval_monic_polynomials(x, n - k)
        )