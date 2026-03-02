import numpy as np
from scipy.linalg import qr, solve_triangular, pinv

def sketch(A, k):
    """
    Sketching A^{T}

    Inputs:
        A:    matrix of size (m, n)
    Outputs:
        V:    row space approximation of A with size (n, k)
    """
    m = A.shape[0]
    Theta = np.random.randn(m, k) / np.sqrt(m)
    V, _ = qr(A.T @ Theta, mode='economic')
    return V


def ARP(A, k):
    """
    Adaptive Randomized Pivoting for CSSP

    Input:
        A:    matrix of size (m, n)
        k:    desired rank
    """
    V = sketch(A, k)
    n = V.shape[0]
    W = V.copy()
    J = []
    for i in range(k):
        prob = []
        for j in range(n):
            prob.append(np.linalg.norm(W[j, :])**2 / (k-i))
           
        jk = np.random.choice(len(prob), p=prob)
        J.append(jk) 

        # update W
        v = W[jk, :]
        W = W @ (np.eye(k) - np.outer(v, v) / np.linalg.norm(v)**2)
 
    return J