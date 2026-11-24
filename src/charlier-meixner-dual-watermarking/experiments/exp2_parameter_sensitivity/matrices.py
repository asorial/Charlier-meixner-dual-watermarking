import numpy as np
from mpmath import mp, mpf, exp, log, loggamma, sqrt, rf, factorial
mp.dps = 60

def charlier_matrix(a, N=8):
    a = mpf(a)
    A = np.zeros((N, N))
    for x in range(N):
        P = [mpf(1), x - a]
        for n in range(1, N - 1):
            P.append((x - (n + a)) * P[n] - n * a * P[n - 1])
        w = exp(-a) * a ** x / factorial(x)
        for n in range(N):
            d = factorial(n) * a ** n
            A[x, n] = float(P[n] * sqrt(w / d))
    return A

def meixner_matrix(m, g, N=8):
    m, g = mpf(m), mpf(g)
    A = np.zeros((N, N))
    for x in range(N):
        b = lambda n: (n * (1 + m) + m * g) / (1 - m)
        c = lambda n: n * m * (n - 1 + g) / (1 - m) ** 2
        P = [mpf(1), x - b(0)]
        for n in range(1, N - 1):
            P.append((x - b(n)) * P[n] - c(n) * P[n - 1])
        w = m ** x * rf(g, x) / factorial(x)
        for n in range(N):
            d = factorial(n) * rf(g, n) * m ** n / (1 - m) ** (g + 2 * n)
            A[x, n] = float(P[n] * sqrt(w / d))
    return A
