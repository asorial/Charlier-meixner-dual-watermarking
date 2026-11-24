
from classes.kernel0jcoeffs_class import Kernel0JCoefficients
from mpmath import *

mp.dps = 25; mp.pretty = True

class Phi(Kernel0JCoefficients):
    def __init__(self, mu, lamb, alpha, beta, i, j, *args):
        Kernel0JCoefficients.__init__(self, *args)
        self.mu = mu
        self.lamb = lamb
        self.alpha = alpha
        self.beta = beta
        self.i = i
        self.j = j

    def __det1__(self, n):
        return (
            self.d_operator_monic_polynomials(self.alpha, n, self.i) *
            (1 + self.mu * self.kernel(self.beta, self.beta, n, self.j, self.j)) -
            self.mu * self.kernel(self.alpha, self.beta, n, self.i, self.j) *
            self.d_operator_monic_polynomials(self.beta, n, self.j)
        )

    def __det2__(self, n):
        return (
            self.d_operator_monic_polynomials(self.beta, n, self.j) *
            (1 + self.lamb * self.kernel(self.alpha, self.alpha, n, self.i, self.i)) -
            self.lamb * self.kernel(self.beta, self.alpha, n, self.j, self.i) *
            self.d_operator_monic_polynomials(self.alpha, n, self.i)
        )

    def __det3__(self, n):
        return (
            (1 + self.lamb * self.kernel(self.alpha, self.alpha, n, self.i, self.i)) *
            (1 + self.mu * self.kernel(self.beta, self.beta, n, self.j, self.j)) -
            self.mu * self.kernel(self.alpha, self.beta, n, self.i, self.j) *
            self.lamb * self.kernel(self.beta, self.alpha, n, self.j, self.i)
        )

    def Phi1(self, n):
        return self.__det1__(n) / self.__det3__(n)

    def Phi2(self, n):
        return self.__det2__(n) / self.__det3__(n)   