
import numpy as np
from mpmath import *

mp.dps = 25; mp.pretty = True

class OrthogonalPolynomials():
    def __init__(self, *args):
        self.args = args

    def beta_n(self, n):
        pass

    def gamma_n(self, n):
        pass

    def monic_polynomials(self, n):
        polyn_0 = [0.0]
        polyn_1 = [1.0]
        if n < 0:
            return polyn_0
        elif n == 0:
            return polyn_1
        else:
            for k in range(n):
                polyn_mod_1 = [0.0, 0.0] + [self.gamma_n(k) * x for x in polyn_0]
                polyn_mod_2 = [0.0] + [self.beta_n(k) * x for x in polyn_1]
                polyn_0 = polyn_1
                polyn_1.append(0.0)
                polyn_1 = [x - y - z for x, y, z in zip(polyn_1, polyn_mod_1, polyn_mod_2)]
            return polyn_1

    def eval_monic_polynomials(self, x, n):
        return np.polyval(self.monic_polynomials(n), x)

    def squared_norm(self, n):
        pass

    def weight_function(self, x):
        pass

    def d_operator_monic_polynomials(self, x, n, k):
        pass

    def kernel(self, x, y, n, i, j):
        return (
            sum(
                [
                    self.d_operator_monic_polynomials(x, k, i) *
                    self.d_operator_monic_polynomials(y, k, j) /
                    self.squared_norm(k)
                    for k in range(n)
                ]
            )
        )