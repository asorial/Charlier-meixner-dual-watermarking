
from classes.phi_class import Phi
from mpmath import *

mp.dps = 25; mp.pretty = True

class ToObtainConnetionFormulaIIM(Phi):
    def __init__(self, mu, lamb, alpha, beta, i, j, *args):
        Phi.__init__(self, mu, lamb, alpha, beta, i, j, *args)

    def Cn(self, x, n):
        return (
            1 -
            self.lamb * self.Phi1(n) * self.An(x, self.alpha, n, self.i) -
            self.mu * self.Phi2(n) * self.An(x, self.beta, n, self.j)
        )

    def Dn(self, x, n):
        return (
            - self.lamb * self.Phi1(n) *
            self.Bn(x, self.alpha, n, self.i) -
            self.mu * self.Phi2(n) *
            self.Bn(x, self.beta, n, self.j)
        )

    def polynomials(self, x, n):
        return self.eval_monic_polynomials(x, n)    