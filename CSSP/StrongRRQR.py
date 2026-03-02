import numpy as np
from scipy.linalg import qr, solve_triangular


def _givens(a, b):
    """Return 2x2 Givens rotation matrix G such that G @ [a, b].T = [r, 0].T."""
    if b == 0:
        c, s = 1.0, 0.0
    elif abs(b) > abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1 + tau * tau)
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / np.sqrt(1 + tau * tau)
        s = c * tau
    return np.array([[c, -s], [s, c]])


def _safe_recip_sqrt(x, R11):
    """
    Compute 1/sqrt(x) element-wise without NaN or Inf.

    For entries where x <= 0 due to floating-point cancellation, fall back to
    an exact recomputation from inv(R11) row norms.

    Parameters
    ----------
    x   : 1-D array  (should be positive, but may not be due to rounding)
    R11 : upper-triangular matrix, shape (k-1, k-1)

    Returns
    -------
    result : 1-D array, same shape as x
    """
    result   = np.empty_like(x)
    good     = x > 0
    result[good] = x[good] ** (-0.5)

    bad_idx = np.where(~good)[0]
    if bad_idx.size == 0:
        return result

    # Exact fallback: recompute from scratch for affected rows only.
    # inv_R11 columns = solve(R11, I), so rows of inv(R11) are accessible.
    inv_R11_rows = solve_triangular(R11, np.eye(R11.shape[0]))  # shape (k-1, k-1)
    for i in bad_idx:
        row_norm = np.linalg.norm(inv_R11_rows[i, :])
        result[i] = 1.0 / row_norm if row_norm > 0 else 0.0

    return result


def sRRQR_rank(A, f, k, truncate=False):
    """
    Strong Rank Revealing QR with fixed rank 'k'.

        A[:, p] = Q @ R = Q @ [[R11, R12],
                                [  0, R22]]

    where inv(R11) @ R12 has all entries bounded in absolute value by f.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Target matrix.
    f : float
        Bounding constant for entries of inv(R11) @ R12.  Must be >= 1.
    k : int
        Desired rank (dimension of R11).

    Parameters (continued)
    ----------------------
    truncate : bool, optional (default True)
        If True,  return Q of shape (m, k) and R of shape (k, n)  [truncated].
        If False, return full Q (m, min(m,n)) and R (min(m,n), n),
        so that A[:, p] = Q @ R holds exactly.

    Returns
    -------
    Q  : ndarray (m, k) if truncate else (m, min(m,n))
    R  : ndarray (k, n) if truncate else (min(m,n), n)
    p  : ndarray (n,)      permutation: A[:, p] = Q @ R
    k  : int               rank actually used
    nb : int               number of column interchanges performed

    Notes
    -----
    Implements Algorithm 4 from:
        Gu & Eisenstat, "Efficient algorithms for computing a strong
        rank-revealing QR factorization", SIAM J. Sci. Comput. 17(4), 1996.
    """

    if f < 1:
        print("f < 1; automatically set to f = 2")
        f = 2.0

    nb    = 0
    m, n  = A.shape
    k     = min(k, m, n)

    # ------------------------------------------------------------------ #
    # Pivoted QR  (most expensive step)                                    #
    # ------------------------------------------------------------------ #
    Q, R, p = qr(A, pivoting=True)      # A[:, p] = Q @ R
    min_mn  = min(m, n)
    Q = Q[:, :min_mn]
    R = R[:min_mn, :]

    if k == n:
        if truncate:
            return Q[:, :k], R[:k, :], p, k, nb
        return Q, R, p, k, nb

    # Make R diagonal entries positive
    ss = np.sign(np.diag(R)) if min(R.shape) > 1 else np.array([np.sign(R[0, 0])])
    R  = R  * ss[:, np.newaxis]
    Q  = Q  * ss[np.newaxis, :]

    Rm = R.shape[0]     # = min(m, n)

    # ------------------------------------------------------------------ #
    # Initialise auxiliary quantities                                       #
    # ------------------------------------------------------------------ #
    # AB[i,j]  = (inv(R11) @ R12)[i,j],  shape (k, n-k)
    AB = solve_triangular(R[:k, :k], R[:k, k:])

    # gamma[j] = || R22[:, j] ||_2,  shape (n-k,)
    gamma = np.linalg.norm(R[k:, k:], axis=0) if k < Rm else np.zeros(n - k)

    # omega[i] = 1 / || row_i of inv(R11) ||_2,  shape (k,)
    inv_R11 = solve_triangular(R[:k, :k], np.eye(k))
    omega   = 1.0 / np.linalg.norm(inv_R11, axis=1)

    # ------------------------------------------------------------------ #
    # Main interchange loop                                                #
    # ------------------------------------------------------------------ #
    while True:
        tmp_mat = np.outer(1.0 / omega, gamma) ** 2 + AB ** 2

        idx = np.argwhere(tmp_mat > f * f)
        if idx.size == 0:
            break
        i, j = idx[0]   # 0-based row i in R11, col j in R22

        # -------------------------------------------------------------- #
        # Step 1: bring column k+j to position k  (0-based)              #
        # -------------------------------------------------------------- #
        if j > 0:
            AB[:, [0, j]]    = AB[:, [j, 0]]
            gamma[[0, j]]    = gamma[[j, 0]]
            R[:, [k, k + j]] = R[:, [k + j, k]]
            p[[k, k + j]]    = p[[k + j, k]]

        # -------------------------------------------------------------- #
        # Step 2: bring column i to position k-1  (0-based)              #
        # -------------------------------------------------------------- #
        if i < k - 1:
            cyc      = list(range(i, k))
            new_cyc  = cyc[1:] + [cyc[0]]
            p[i:k]      = p[new_cyc]
            R[:, i:k]   = R[:, new_cyc]
            omega[i:k]  = omega[new_cyc]
            AB[i:k, :]  = AB[new_cyc, :]

            for ii in range(i, k - 1):
                G = _givens(R[ii, ii], R[ii + 1, ii])
                if G[0, :] @ R[ii:ii + 2, ii] < 0:
                    G = -G
                R[ii:ii + 2, :]  = G @ R[ii:ii + 2, :]
                Q[:, ii:ii + 2]  = Q[:, ii:ii + 2] @ G.T

            if R[k - 1, k - 1] < 0:
                R[k - 1, :] = -R[k - 1, :]
                Q[:, k - 1] = -Q[:, k - 1]

        # -------------------------------------------------------------- #
        # Step 3: zero sub-diagonal of column k  (0-based)               #
        # -------------------------------------------------------------- #
        if k < Rm:
            for ii in range(k + 1, Rm):
                G = _givens(R[k, k], R[ii, k])
                if G[0, :] @ R[[k, ii], k] < 0:
                    G = -G
                R[[k, ii], :]  = G @ R[[k, ii], :]
                Q[:, [k, ii]]  = Q[:, [k, ii]] @ G.T

        # -------------------------------------------------------------- #
        # Step 4: swap columns k-1 and k  (0-based)                       #
        # -------------------------------------------------------------- #
        nb += 1
        p[[k - 1, k]] = p[[k, k - 1]]

        ga      = R[k - 1, k - 1]
        mu      = R[k - 1, k] / ga
        nu      = R[k, k] / ga if k < Rm else 0.0
        rho     = np.hypot(mu, nu)
        ga_bar  = ga * rho

        b1  = R[:k - 1, k - 1].copy()
        b2  = R[:k - 1, k].copy()
        c1T = R[k - 1, k + 1:].copy()
        c2T = R[k, k + 1:].copy() if k < Rm else np.zeros(n - k - 1)

        c1T_bar = (mu * c1T + nu * c2T) / rho
        c2T_bar = (nu * c1T - mu * c2T) / rho

        # Update R
        R[:k - 1, k - 1]  = b2
        R[:k - 1, k]       = b1
        R[k - 1, k - 1]    = ga_bar
        R[k - 1, k]        = ga * mu / rho
        if k < Rm:
            R[k, k]         = ga * nu / rho
        R[k - 1, k + 1:]   = c1T_bar
        if k < Rm:
            R[k, k + 1:]    = c2T_bar

        # Update AB
        # Note: after the swap above, R[:k-1, :k-1] == old R11[0:k-1, 0:k-1]
        u  = solve_triangular(R[:k - 1, :k - 1], b1)
        u1 = AB[:k - 1, 0].copy()
        AB[:k - 1, 0]  = (nu * nu * u - mu * u1) / (rho * rho)
        AB[k - 1, 0]   = mu / (rho * rho)
        AB[k - 1, 1:]  = c1T_bar / ga_bar
        AB[:k - 1, 1:] += (np.outer(nu * u, c2T_bar)
                           - np.outer(u1, c1T_bar)) / ga_bar

        # Update gamma
        gamma[0]  = ga * nu / rho
        gamma[1:] = np.sqrt(np.maximum(
            gamma[1:] ** 2 + c2T_bar ** 2 - c2T ** 2, 0.0
        ))

        # Update omega via rank-1 formula; fall back to exact recomputation
        # for any entry that goes non-positive due to floating-point cancellation.
        u_bar        = u1 + mu * u
        omega[k - 1] = ga_bar
        denom_sq     = (omega[:k - 1] ** (-2)
                        + u_bar ** 2 / (ga_bar ** 2)
                        - u ** 2 / (ga ** 2))
        omega[:k - 1] = _safe_recip_sqrt(denom_sq, R11=R[:k - 1, :k - 1])

        # Apply 2x2 Givens-like rotation to Q
        Gk = np.array([[mu / rho,  nu / rho],
                       [nu / rho, -mu / rho]])
        if k < Rm:
            Q[:, [k - 1, k]] = Q[:, [k - 1, k]] @ Gk.T

    if truncate:
        return Q[:, :k], R[:k, :], p, k, nb
    return Q, R, p, k, nb

def sRRQR_tol(A, f, tol, truncate=False):
    """
    Strong Rank Revealing QR with an error threshold tol.

        A[:, p] = Q @ R = Q @ [[R11, R12],
                                [  0, R22]]

    where inv(R11) @ R12 has all entries bounded by f, and every column of
    R22 has norm less than tol.

    Parameters
    ----------
    A        : ndarray, shape (m, n)
        Target matrix.
    f        : float
        Bounding constant for entries of inv(R11) @ R12.  Must be >= 1.
    tol      : float
        Threshold: the rank k is chosen so that all singular values of the
        tail R22 are below tol (equivalently, diag(R)[k:] < tol after
        pivoted QR).
    truncate : bool, optional (default True)
        If True,  return Q (m, k) and R (k, n).
        If False, return full Q (m, min(m,n)) and R (min(m,n), n).

    Returns
    -------
    Q  : ndarray  orthogonal factor
    R  : ndarray  upper factor
    p  : ndarray (n,)   permutation: A[:, p] = Q_full @ R_full
    k  : int            rank chosen to meet tol
    nb : int            number of column interchanges

    Notes
    -----
    Implements Algorithm 5 from:
        Gu & Eisenstat, "Efficient algorithms for computing a strong
        rank-revealing QR factorization", SIAM J. Sci. Comput. 17(4), 1996.
    """

    if f < 1:
        print("f < 1; automatically set to f = 2")
        f = 2.0

    nb    = 0
    m, n  = A.shape

    # ------------------------------------------------------------------ #
    # Pivoted QR                                                           #
    # ------------------------------------------------------------------ #
    Q, R, p = qr(A, pivoting=True)
    min_mn  = min(m, n)
    Q = Q[:, :min_mn]
    R = R[:min_mn, :]

    # Make R diagonal entries positive
    ss = np.sign(np.diag(R)) if min(R.shape) > 1 else np.array([np.sign(R[0, 0])])
    R  = R * ss[:, np.newaxis]
    Q  = Q * ss[np.newaxis, :]

    Rm = R.shape[0]   # = min(m, n)

    # ------------------------------------------------------------------ #
    # Determine initial rank k from tolerance                             #
    # ------------------------------------------------------------------ #
    diag_R = np.diag(R)
    above  = np.where(diag_R > tol)[0]

    # Special case: every diagonal entry is below tol -> rank 0
    if above.size == 0:
        if truncate:
            return np.zeros((m, 0)), np.zeros((0, n)), np.arange(n), 0, nb
        return Q, R, p, 0, nb

    k = int(above[-1]) + 1   # number of diagonal entries above tol (1-based count)

    # Special case: rank equals number of columns
    if k == n:
        print("Rank equals the number of columns!")
        if truncate:
            return Q[:, :k], R[:k, :], p, k, nb
        return Q, R, p, k, nb

    # ------------------------------------------------------------------ #
    # Initialise auxiliary quantities                                      #
    # ------------------------------------------------------------------ #
    AB    = solve_triangular(R[:k, :k], R[:k, k:])
    gamma = np.linalg.norm(R[k:, k:], axis=0) if k < Rm else np.zeros(n - k)
    inv_R11 = solve_triangular(R[:k, :k], np.eye(k))
    omega   = 1.0 / np.linalg.norm(inv_R11, axis=1)

    # ------------------------------------------------------------------ #
    # Main outer loop                                                      #
    # ------------------------------------------------------------------ #
    while True:

        # ============================================================== #
        # STEP 1: Strong RRQR with fixed rank k                           #
        # ============================================================== #
        while True:
            tmp_mat = np.outer(1.0 / omega, gamma) ** 2 + AB ** 2
            idx     = np.argwhere(tmp_mat > f * f)
            if idx.size == 0:
                break
            i, j = idx[0]   # 0-based

            # Step 1a: bring column k+j to position k  (0-based)
            if j > 0:
                AB[:, [0, j]]    = AB[:, [j, 0]]
                gamma[[0, j]]    = gamma[[j, 0]]
                R[:, [k, k + j]] = R[:, [k + j, k]]
                p[[k, k + j]]    = p[[k + j, k]]

            # Step 1b: bring column i to position k-1  (0-based)
            if i < k - 1:
                cyc     = list(range(i, k))
                new_cyc = cyc[1:] + [cyc[0]]
                p[i:k]      = p[new_cyc]
                R[:, i:k]   = R[:, new_cyc]
                omega[i:k]  = omega[new_cyc]
                AB[i:k, :]  = AB[new_cyc, :]

                for ii in range(i, k - 1):
                    G = _givens(R[ii, ii], R[ii + 1, ii])
                    if G[0, :] @ R[ii:ii + 2, ii] < 0:
                        G = -G
                    R[ii:ii + 2, :]  = G @ R[ii:ii + 2, :]
                    Q[:, ii:ii + 2]  = Q[:, ii:ii + 2] @ G.T

                if R[k - 1, k - 1] < 0:
                    R[k - 1, :] = -R[k - 1, :]
                    Q[:, k - 1] = -Q[:, k - 1]

            # Step 1c: zero sub-diagonal of column k  (0-based)
            if k < Rm:
                for ii in range(k + 1, Rm):
                    G = _givens(R[k, k], R[ii, k])
                    if G[0, :] @ R[[k, ii], k] < 0:
                        G = -G
                    R[[k, ii], :]  = G @ R[[k, ii], :]
                    Q[:, [k, ii]]  = Q[:, [k, ii]] @ G.T

            # Step 1d: swap columns k-1 and k  (0-based)
            nb += 1
            p[[k - 1, k]] = p[[k, k - 1]]

            ga     = R[k - 1, k - 1]
            mu     = R[k - 1, k] / ga
            nu     = R[k, k] / ga if k < Rm else 0.0
            rho    = np.hypot(mu, nu)
            ga_bar = ga * rho

            b1  = R[:k - 1, k - 1].copy()
            b2  = R[:k - 1, k].copy()
            c1T = R[k - 1, k + 1:].copy()
            c2T = R[k, k + 1:].copy() if k < Rm else np.zeros(n - k - 1)

            c1T_bar = (mu * c1T + nu * c2T) / rho
            c2T_bar = (nu * c1T - mu * c2T) / rho

            # Update R
            R[:k - 1, k - 1]  = b2
            R[:k - 1, k]       = b1
            R[k - 1, k - 1]    = ga_bar
            R[k - 1, k]        = ga * mu / rho
            if k < Rm:
                R[k, k]         = ga * nu / rho
            R[k - 1, k + 1:]   = c1T_bar
            if k < Rm:
                R[k, k + 1:]    = c2T_bar

            # Update AB
            u  = solve_triangular(R[:k - 1, :k - 1], b1)
            u1 = AB[:k - 1, 0].copy()
            AB[:k - 1, 0]  = (nu * nu * u - mu * u1) / (rho * rho)
            AB[k - 1, 0]   = mu / (rho * rho)
            AB[k - 1, 1:]  = c1T_bar / ga_bar
            AB[:k - 1, 1:] += (np.outer(nu * u, c2T_bar)
                               - np.outer(u1, c1T_bar)) / ga_bar

            # Update gamma
            gamma[0]  = ga * nu / rho
            gamma[1:] = np.sqrt(np.maximum(
                gamma[1:] ** 2 + c2T_bar ** 2 - c2T ** 2, 0.0
            ))

            # Update omega (with exact fallback for numerical cancellation)
            u_bar        = u1 + mu * u
            omega[k - 1] = ga_bar
            denom_sq     = (omega[:k - 1] ** (-2)
                            + u_bar ** 2 / (ga_bar ** 2)
                            - u ** 2 / (ga ** 2))
            omega[:k - 1] = _safe_recip_sqrt(denom_sq, R11=R[:k - 1, :k - 1])

            # Apply 2x2 rotation to Q
            Gk = np.array([[mu / rho,  nu / rho],
                           [nu / rho, -mu / rho]])
            if k < Rm:
                Q[:, [k - 1, k]] = Q[:, [k - 1, k]] @ Gk.T

        # ============================================================== #
        # STEP 2: Try to reduce rank k -> k-1 if min(omega) <= tol       #
        # ============================================================== #
        i_min  = int(np.argmin(omega))
        min_om = omega[i_min]

        # min(omega) > tol means rank-(k-1) cannot meet threshold: done
        if min_om > tol:
            break

        # Bring the weakest column (i_min) to position k-1 (0-based)
        if i_min < k - 1:
            cyc     = list(range(i_min, k))
            new_cyc = cyc[1:] + [cyc[0]]
            p[i_min:k]     = p[new_cyc]
            R[:, i_min:k]  = R[:, new_cyc]

            for ii in range(i_min, k - 1):
                G = _givens(R[ii, ii], R[ii + 1, ii])
                if G[0, :] @ R[ii:ii + 2, ii] < 0:
                    G = -G
                R[ii:ii + 2, :]  = G @ R[ii:ii + 2, :]
                Q[:, ii:ii + 2]  = Q[:, ii:ii + 2] @ G.T

            if R[k - 1, k - 1] < 0:
                R[k - 1, :] = -R[k - 1, :]
                Q[:, k - 1] = -Q[:, k - 1]

        # Reduce rank
        k -= 1

        if k == 0:
            break

        # Recompute AB, gamma, omega from scratch at new rank k
        # (cheap relative to the overall algorithm, and avoids drift)
        AB      = solve_triangular(R[:k, :k], R[:k, k:])
        gamma   = np.linalg.norm(R[k:, k:], axis=0) if k < Rm else np.zeros(n - k)
        inv_R11 = solve_triangular(R[:k, :k], np.eye(k))
        omega   = 1.0 / np.linalg.norm(inv_R11, axis=1)

    if truncate:
        return Q[:, :k], R[:k, :], p, k, nb
    return Q, R, p, k, nb



def sRRQR(A, f, type_, par,truncate = False):
    """
    Strong Rank Revealing QR (SRRQR)
    
    A P = [Q1, Q2] * [R11, R12;
                       0,  R22]
    which satisfies that inv(R11) * R12 has entries 
    bounded by a pre-specified constant f (>= 1).
    
    Parameters
    ----------
    A : ndarray
        Target matrix to be approximated.
    f : float
        Constant that bounds entries of inv(R11) * R12. Must be >= 1.
    type_ : str
        Either 'rank' or 'tol'.
        - 'rank': fix the rank (Algorithm 4 in Gu & Eisenstat 1996).
        - 'tol' : fix the error tolerance (Algorithm 6 in Gu & Eisenstat 1996).
    par : int or float
        Parameter depending on type_:
        - if type_ == 'rank': par = k (rank).
        - if type_ == 'tol' : par = tol (error threshold).
    
    Returns
    -------
    Q : ndarray
        Orthonormal matrix Q1, shape (m, k).
    R : ndarray
        Upper triangular matrix [R11, R12], shape (k, n).
    p : ndarray
        Permutation.
    k : int
        Effective rank chosen.
    nb : int
        Number of interchanges performed.
    
    Reference
    ---------
    Gu, Ming, and Stanley C. Eisenstat. 
    "Efficient algorithms for computing a strong rank-revealing QR factorization." 
    SIAM Journal on Scientific Computing 17.4 (1996): 848-869.
    """

    if type_ == "rank":
        Q, R, p, k, nb = sRRQR_rank(A, f, par,truncate)
        return Q, R, p, k, nb

    elif type_ == "tol":
        Q, R, p, k, nb = sRRQR_tol(A, f, par,truncate)
        return Q, R, p, k, nb

    else:
        raise ValueError(f"Unknown type '{type_}', must be 'rank' or 'tol'.")

