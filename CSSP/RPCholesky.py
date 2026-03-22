import numpy as np
from scipy.linalg import qr, solve_triangular, pinv, norm

def RPCholesky(A, k):
    """
    Randomly pivoted Cholesky for CSSP

    Input:
        A:      matrix of size (m, n)
        k:      desired rank

    Output:
        J:      ndarray (n,)   permutation
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
    return J, k


def RPCholesky_tol(A, tol):
    """
    Randomly pivoted Cholesky for CSSP with tolerence

    Input:
        A:      matrix of size (m, n)
        tol:    Threshold: the rank k is chosen so that

    Output:
        J:      ndarray (n,)   permutation
    """

    m, n = A.shape
    M = A.T @ A
    F = np.zeros((n, n))
    d = np.diag(M)
    J = []
    k = 0
    while k < n:
        prob = []
        prob = d / d.sum()
        jk = np.random.choice(len(prob), p=prob)
        J.append(jk)
        g = M[:, jk]
        g = g - F[:, :k] @ F[jk, :k].T
        F[:, k] = g / np.sqrt(g[jk])
        d = d - F[:, k]**2
        d[d < 0] = 0
        if norm(M - F[:, :k+1] @ F[:, :k+1].T, 'fro') / norm(M, 'fro') < tol:
            break
        k += 1

    if k == n - 1:
        print("Rank equals the number of columns!")

    return J, k+1