import numpy as np
from scipy.linalg import qr, solve_triangular, pinv

def RPCholesky(A, k):
    """
    Randomized Pivoting for CSSP based on Randomly pivoted Cholesky

    Input:
        A:    psd matrix of size (m, n)
        k:    desired rank
    """
    m, n = A.shape
    M = A.T @ A  # size (n, n)
    F = np.zeros((n, k))
    d = np.diag(M)
    J = []
    for i in range(k):
        prob = []
        prob = d / d.sum()
        jk = np.random.choice(len(prob), p=prob)
        J.append(jk)
        g = M[:, jk]
        g = g - F[:, :i] @ F[jk, :i].T
        F[:, i] = g / np.sqrt(g[jk])
        d = d - F[:, i]**2
        d[d < 0] = 0   
    return J