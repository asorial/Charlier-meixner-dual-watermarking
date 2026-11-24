
import numpy as np
from mpmath import *

mp.dps = 25; mp.pretty = True

class PolynomialsZeros():
    def __init__(self) -> None:
        pass

    def polynomials_zeros(self, L):
        return sorted(list(np.roots(L)))