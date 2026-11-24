

from helpers.blocks_class import BlocksImage
import numpy as np 

class Block32Matrix():
    def __init__(self, block):
        self.matrix = block

    def get_matrix(self):
        return self.matrix

    def get_list(self):
        L_1 = [0, 3, 12, 15]
        L = []
        matrix_8x8 = BlocksImage(self.get_matrix())
        n = matrix_8x8.max_num_blocks()
        for i in range(n):
            if i not in L_1:
                Laux = list(matrix_8x8.get_block(i).reshape(-1))
                L.extend(Laux) 
        return L

    def get_elem(self, num_elem):
        return self.get_list()[num_elem]

    def set_elem(self, elem, num_elem):
        j = 0
        L = self.get_list()
        L[num_elem] = elem
        L_1 = [0, 3, 12, 15]
        matrix_8x8 = BlocksImage(self.get_matrix())
        n = matrix_8x8.max_num_blocks()
        for i in range(n):
            if i not in L_1:
                __list = np.array(L[64 * j: (j + 1) * 64])
                block = __list.reshape(8, 8)
                matrix_8x8.set_block(block, i)
                j += 1

    def set_list(self, L):
        j = 0
        L_1 = [0, 3, 12, 15]
        matrix_8x8 = BlocksImage(self.get_matrix())
        n = matrix_8x8.max_num_blocks()
        for i in range(n):
            if i not in L_1:
                __list = np.array(L[64 * j: (j + 1) * 64])
                block = __list.reshape(8, 8)
                matrix_8x8.set_block(block, i)
                j += 1
