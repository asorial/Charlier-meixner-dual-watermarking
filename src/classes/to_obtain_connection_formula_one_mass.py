
from classes.kernel0jcoeffs_class import Kernel0JCoefficients
from mpmath import *

mp.dps = 25; mp.pretty = True

class ToObtainConnetionFormulaIM(Kernel0JCoefficients):
    def __init__(self, lamb, alpha, j, *args):
        Kernel0JCoefficients.__init__(self, *args)
        self.lamb = lamb
        self.alpha = alpha
        self.j = j

    def Cn(self, x, n):
        return (
            1 -
            self.lamb * 
            self.d_operator_monic_polynomials(self.alpha, n, self.j) *
            self.An(x, self.alpha, n, self.j) /
            (1 + self.lamb * self.kernel(self.alpha, self.alpha, n, self.j, self.j))
        )

    def Dn(self, x, n):
        return (
            - self.lamb * 
            self.d_operator_monic_polynomials(self.alpha, n, self.j) *
            self.Bn(x, self.alpha, n, self.j) /
            (1 + self.lamb * self.kernel(self.alpha, self.alpha, n, self.j, self.j))
        )

    def polynomials(self, x, n):
        return self.eval_monic_polynomials(x, n) 