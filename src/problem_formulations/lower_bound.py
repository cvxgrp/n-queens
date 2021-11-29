
import numpy as np
import scipy.sparse as sp

def construct_A(n):
    A = sp.dok_matrix((6*n - 1, 4*n**2 + 4*n))

    x_index = np.arange(4 * n**2 + 4 * n)
    W_idx, N_idx, E_idx, S_idx, anti_diag_terms, diag_terms = vec_to_matrices(x_index, n)
    
    idx_to_A = 0
    diag_index = 0
    anti_diag_index = 0

    for i in range(1, n):
        A[idx_to_A, W_idx[i,:]] = 1
        A[idx_to_A, N_idx[i,:]] = 1
        A[idx_to_A, E_idx[i,:]] = 1
        A[idx_to_A, S_idx[i,:]] = 1
        idx_to_A += 1
    
    for j in range(n):
        A[idx_to_A, W_idx[:,j]] = 1
        A[idx_to_A, N_idx[:,j]] = 1
        A[idx_to_A, E_idx[:,j]] = 1
        A[idx_to_A, S_idx[:,j]] = 1
        idx_to_A += 1

    #  Diagonal sums.
    # There are 2n diagonals, starting from each column and each row.
    # The first diagonal starting from column 0 is N_{00}, E_{00}, W_{01}, S_{01}, N_{1,1}, E_{11}, … , N_{n-1,n-1}, E_{n-1,n-1}.
    # The diagonal associated with the last column is N_{0,n-1}+E_{0,n-1}.

    for i in range(n-1, -n-1, -1):
        A[idx_to_A, np.diag(np.fliplr(N_idx), k=i)] = 1
        A[idx_to_A, np.diag(np.fliplr(W_idx), k=i)] = 1
        A[idx_to_A, np.diag(np.fliplr(S_idx), k=i+1)] = 1
        A[idx_to_A, np.diag(np.fliplr(E_idx), k=i+1)] = 1
        A[idx_to_A, anti_diag_terms[anti_diag_index]] = 1
        idx_to_A += 1
        anti_diag_index += 1

    for i in range(-n, n):
        A[idx_to_A, np.diag(N_idx, k=i)] = 1
        A[idx_to_A, np.diag(E_idx, k=i)] = 1
        A[idx_to_A, np.diag(S_idx, k=i+1)] = 1
        A[idx_to_A, np.diag(W_idx, k=i+1)] = 1
        A[idx_to_A, diag_terms[diag_index]] = 1
        idx_to_A += 1
        diag_index += 1

    b = np.ones(6*n - 1) / n
    return A.tocsr(), b

def construct_objective_fxn_derivs(n):
    def f(x):
        return (x * np.log(x)).sum() + 4 * np.log(n) + 2 * np.log(2) + 3

    def df(x):
        return np.log(x) + 1
    def d2f(x):
        return 1 / x
    return f, df, d2f

def vec_to_matrices(x, n):
    n_squared = n**2
    assert x.shape == (4 * n_squared + 4 * n,)
    W = x[:n_squared].reshape((n, n))
    N = x[n_squared:2 * n_squared].reshape((n, n))
    E = x[2 * n_squared:3 * n_squared].reshape((n, n))
    S = x[3 * n_squared:4 * n_squared].reshape((n, n))

    anti_diag_terms = x[4 * n_squared: 4 * n_squared + 2 * n]
    diag_terms = x[4 * n_squared + 2 * n: 4 * n_squared + 4 * n]
    return W, N, E, S, anti_diag_terms, diag_terms

def calculate_lagrangian(A, b, n, v):
    vTb_term  = v.T @ b
    neg_f_star_term = -np.exp(A.T @ v - 1).sum() + 4 * np.log(n) + 2 * np.log(2) + 3
    return  vTb_term + neg_f_star_term
