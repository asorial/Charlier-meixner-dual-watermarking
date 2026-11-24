
from mpmath import *

mp.dps = 25; mp.pretty = True

class SobolevTypePolynomials():
    def __init__(self) -> None:
        pass

    def Cn(self, x, n):
        pass

    def Dn(self, x, n):
        pass

    def polynomials(self, x, n):
        pass

    def eval_monic_sobolev_type_polynomials(self, x, n):
        return (
            self.Cn(x, n) *
            self.polynomials(x, n) +
            self.Dn(x, n) *
            self.polynomials(x, n - 1)
        )

    def monic_sobolev_type_polynomials(self, n):
        return list(
                reversed(taylor(lambda x: self.eval_monic_sobolev_type_polynomials(x, n), 0, n))
            )