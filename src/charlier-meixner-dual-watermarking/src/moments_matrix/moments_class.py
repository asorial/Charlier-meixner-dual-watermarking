
import numpy as np

class SeparableMoments():
    def __init__(self, left_orthon_matrix, right_orthon_matrix):
        self.l_orthon_matrix = left_orthon_matrix
        self.r_orthon_matrix = right_orthon_matrix

    def direct_moments(self, A):
        return self.l_orthon_matrix.T @ A @ self.r_orthon_matrix

    def inverse_moments(self, A):
        matrix = self.l_orthon_matrix @ A @ self.r_orthon_matrix.T
        return np.clip(np.rint(matrix), 0, 255)


class SeparableFractionalMoments():
    def __init__(self, lmatrix_d, rmatrix_d, lmatrix_i, rmatrix_i):
        self.lmatrix_d = lmatrix_d
        self.rmatrix_d = rmatrix_d
        self.lmatrix_i = lmatrix_i
        self.rmatrix_i = rmatrix_i

    def direct_moments(self, A):
        return self.lmatrix_d @ A @ self.rmatrix_d

    def inverse_moments(self, A):
        matrix = self.lmatrix_i @ A @ self.rmatrix_i
        return np.clip(np.rint(matrix), 0, 255)
