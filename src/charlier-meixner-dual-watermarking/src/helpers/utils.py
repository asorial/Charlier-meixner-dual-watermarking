# -*- coding: utf-8 -*-

from helpers.dat import DAT
import numpy as np
import hashlib
import math


def sha256_to_key_bin(key):
    hexa_data = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return "".join(format(ord(x), '08b') for x in hexa_data)[:256]

def sha512_to_key_bin(key):
    hexa_data = hashlib.sha512(key.encode('utf-8')).hexdigest()
    return "".join(format(ord(x), '08b') for x in hexa_data)[:512]


def sha256_to_image_bin(imag):
    hexa_data = hashlib.sha256(imag).hexdigest()
    return "".join(format(ord(x), '08b') for x in hexa_data)[:256]


def blake2b_bin(key):
    hexa_data = hashlib.blake2b(key.encode('utf-8')).hexdigest()
    return "".join(format(ord(x), '08b') for x in hexa_data)


def pwlcm(x, p):
    if x >= 0 and x < p:
        return x / p
    elif x >= p and x < 0.5:
        return (x - p) / (0.5 - p)
    elif x >= 0.5 and x < 1:
        return pwlcm(1 - x, p)


def chaotic_map(x, p, n):
    L = []
    for i in range(n):
        x = pwlcm(x, p)
        L.append(int(math.floor(x * 10 ** (14) % n)))
    return L


def perm(L, ind):
    pos = []
    n = len(ind)
    for i in range(n):
        pos.append(L[ind[i]])
    return pos


def set_diff(L1, L2):
    n = len(L2)
    for i in range(n):
        L1.remove(L2[i])
    return L1


def list_reduced(L):
    R = []
    for i in L:
        if i not in R:
            R.append(i)
    return R


def random_list(x, p, L):
    pos = []
    ind = []
    ind = list_reduced(chaotic_map(x, p, len(L)))
    pos = perm(L, ind)
    if len(pos) == len(L):
        return pos
    elif len(pos) == 1:
        return L
    else:
        return pos + random_list(x, p, set_diff(L, pos))
    return pos


def replace(byte_init, bit):
    if bit == '0':
        if byte_init % 2 == 0:
            byte_fin = byte_init
        else:
            byte_fin = byte_init - 1
    elif bit == '1':
        if byte_init % 2 == 0:
            byte_fin = byte_init + 1
        else:
            byte_fin = byte_init
    return byte_fin


def ext_lsb(byte):
    if byte % 2 == 0:
        return '0'
    return '1'


def or_operation(x, y):
    if len(x) != len(y):
        raise ValueError("It is not possible to perform this operation")
    n = len(x)
    return "".join(("0" if x[i] == y[i] else "1") for i in range(n))


def increase(key, n):
    seq = blake2b_bin(key)
    while len(seq) < n:
        seq += blake2b_bin(seq[-128:])
    return seq[:n]


def bin2dec(bin_seq):
    dec_repr = 0
    for i in range(len(bin_seq)):
        if bin_seq[len(bin_seq) - i - 1] != '0':
            dec_repr += 2 ** i
    return dec_repr


def char2bin(data):
    return "".join(format(ord(x), '08b') for x in data)


def bin2char(bin_seq):
    return ''.join(
        (chr(int(bin_seq[i:i+8], 2)) for i in range(0, len(bin_seq), 8))
    )


def list_pos(cad):
    return [i for i in range(len(cad)) if cad[i] == "1"]


def permutation(cad):
    L = [i for i in range(len(cad)) if cad[i] == "1"]
    L.extend([i for i in range(len(cad)) if cad[i] == "0"])
    return L


def parti_bin_msg(bin_msg, m=8):
    n = len(bin_msg) // m
    return [bin_msg[i * m:(i + 1) * m] for i in range(n)]


def increase_msg(msg, n):
    seq = msg
    while len(seq) < n:
        seq += msg
    return seq[:n]


def lsb(byte):
    if byte % 2 == 0:
        return "0"
    else:
        return "1"


def bitxor(bit_1, bit_2):
    if bit_1 == bit_2:
        return "0"
    else:
        return "1"


def matrix_zig_zag():
    A = np.array([
        [0,    1,     8,    16,     9,     2,     3,    10],
        [17,    24,    32,    25,    18,    11,     4,     5],
        [12,    19,    26,    33,    40,    48,    41,    34],
        [27,    20,    13,     6,     7,    14,    21,    28],
        [35,    42,    49,    56,    57,    50,    43,    36],
        [29,    22,    15,    23,    30,    37,    44,    51],
        [58,    59,    52,    45,    38,    31,    39,    46],
        [53,    60,    61,    54,    47,    55,    62,    63]])
    return A


def vzig_zag_scan(A):
    L = []
    permutation = matrix_zig_zag().reshape(-1)
    seq = A.reshape(-1)
    for pos in permutation:
        L.append(seq[pos])
    return L


def inv_vzig_zag_scan(A):
    return vzig_zag_scan(A.T)


def mzig_zag_scan(vect):
    L = list(np.zeros(len(vect)))
    positions = list(matrix_zig_zag().reshape(-1))
    for index, pos in enumerate(positions):
        L[pos] = vect[index]
    return np.array(L).reshape(8, 8)


def quant(A, QF):
    if QF > 50:
        F = (100 - QF) / 50
    else:
        F = 50 / QF
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])
    vect = Q.reshape(-1)
    vect = [vect[item] * F if item in [1, 8, 16] else 1 for item in range(len(vect))] 
    quant_matrix = [i / j for i, j in zip(A.reshape(-1), vect)]
    return np.array(quant_matrix).reshape(8, 8)


def quantinv(A, QF):
    if QF > 50:
        F = (100 - QF) / 50
    else:
        F = 50 / QF
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])
    vect = Q.reshape(-1)
    vect = [vect[item] * F if item in [1, 8, 16] else 1 for item in range(len(vect))]
    quant_matrix = [i * j for i, j in zip(A.reshape(-1), vect)]
    return np.array(quant_matrix).reshape(8, 8)


def arnold_transform(watermark_array, n):
    for item in range(n):
        watermark_array = DAT().dat2(watermark_array)
    return watermark_array


def binarize_watermark(L):
    return "".join(
        str(bit) for bit in [1 if elem == 255 else elem for elem in L]
    )


def binarize_watermark_int(L):
    return [1 if elem == 255 else 0 for elem in L]


def unbinarize_watermark(L):
    return [255 if elem == 1 else elem for elem in L]


def cropping_noise(B, m=8, n=8):
    if len(B.shape) == 2:
        for i in range(m):
            for j in range(n):
                B[i][j] = 0
    else:
        for i in range(m):
            for j in range(n):
                B[i][j][:] = 0
    return B


def remove_item_list(L, item):
    L.pop(item)
    return(L)


def dct_orthogonal_matrix():
    return np.array([
            [0.353553, 0.490393, 0.46194, 0.415735, 0.353553, 0.277785,
                0.191342, 0.0975452],
            [0.353553, 0.415735, 0.191342, -0.0975452, -0.353553, -0.490393,
                -0.46194, -0.277785],
            [0.353553, 0.277785, -0.191342, -0.490393, -0.353553, 0.0975452,
                0.46194, 0.415735],
            [0.353553, 0.0975452, -0.46194, -0.277785, 0.353553, 0.415735,
                -0.191342, -0.490393],
            [0.353553, -0.0975452, -0.46194, 0.277785, 0.353553, -0.415735,
                -0.191342, 0.490393],
            [0.353553, -0.277785, -0.191342, 0.490393, -0.353553, -0.0975452,
                0.46194, -0.415735],
            [0.353553, -0.415735, 0.191342, 0.0975452, -0.353553, 0.490393,
                -0.46194, 0.277785],
            [0.353553, -0.490393, 0.46194, -0.415735, 0.353553, -0.277785,
                0.191342, -0.0975452]])


def mean_signal(matrix_signal):
    values = []
    for item in range(len(matrix_signal[0])):
        values.append([signal[item] for signal in matrix_signal])
    return [np.mean(nsignal) for nsignal in values]