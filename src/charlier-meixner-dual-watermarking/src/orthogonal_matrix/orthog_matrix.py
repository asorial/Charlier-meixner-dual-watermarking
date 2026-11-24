
from weighted_polynomials.weighted_polyn_class import WeightedPolynomials
import numpy as np
from mpmath import *

mp.dps = 25; mp.pretty = True


class OrthogonalMatrix(WeightedPolynomials):
    def __init__(self, size_block, *args):
        WeightedPolynomials.__init__(self, *args)
        self.sblock = size_block

    def __get_row__(self, i):
        return np.array(
            [self.weighted_polyn(i, j) for j in range(self.sblock)]
        )

    def orthog_matrix(self):
        matrix = np.empty((self.sblock, self.sblock))
        for i in range(self.sblock):
            matrix[i, :] = self.__get_row__(i)
        return matrix