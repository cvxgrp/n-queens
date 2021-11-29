
import numpy as np
import scipy.sparse

def vec_to_matrices(x, n):
    n_squared = n**2
    assert x.shape == (4 * n_squared + 4 * n,)
    N = x[:n_squared].reshape((n, n))
    E = x[n_squared:2 * n_squared].reshape((n, n))
    S = x[2 * n_squared:3 * n_squared].reshape((n, n))
    W = x[3 * n_squared:4 * n_squared].reshape((n, n))

    diag_terms = x[4 * n_squared: 4 * n_squared + 2 * n]
    anti_diag_terms = x[4 * n_squared + 2 * n: 4 * n_squared + 4 * n]
    return N, E, S, W, diag_terms, anti_diag_terms

def matrices_to_vec(*matrices):
    return np.hstack([m.reshape((-1,)) for m in matrices])


def construct_A(n):
    A = scipy.sparse.dok_matrix((10 * n, 4* n**2 + 4 * n))
    x_index = np.arange(4 * n**2 + 4 * n)
    N_idx, E_idx, S_idx, W_idx, diag_terms, anti_diag_terms = vec_to_matrices(x_index, n)

    idx_to_A = 0
    diag_index = 0
    anti_diag_index = 0
    b = []

    # SigN_i = n
    for i in range(n):
        A[idx_to_A, N_idx[i,:]] = 1
        b.append(n)
        idx_to_A += 1

    # SigE_j = n
    for j in range(n):
        A[idx_to_A, E_idx[:,j]] = 1
        b.append(n)
        idx_to_A += 1

    # SigS_i = n
    for i in range(n):
        A[idx_to_A, S_idx[i,:]] = 1
        b.append(n)
        idx_to_A += 1

    # SigW_j = n
    for j in range(n):
        A[idx_to_A, W_idx[:,j]] = 1
        b.append(n)
        idx_to_A += 1


    # SigNS_i = n
    for j in range(n):
        A[idx_to_A, N_idx[:,j]] = 1
        A[idx_to_A, S_idx[:,j]] = 1
        b.append(2 * n)
        idx_to_A += 1

    # SigEW_j = n
    for i in range(n):
        A[idx_to_A, E_idx[i,:]] = 1
        A[idx_to_A, W_idx[i,:]] = 1
        b.append(2 * n)
        idx_to_A += 1

    #  Diagonal sums.
    # There are 2n diagonals, starting from each column and each row.
    # The first diagonal starting from column 0 is N_{00}, E_{00}, W_{01}, S_{01}, N_{1,1}, E_{11}, … , N_{n-1,n-1}, E_{n-1,n-1}.
    # The diagonal associated with the last column is N_{0,n-1}+E_{0,n-1}.
    for i in range(-n, n):
        A[idx_to_A, np.diag(N_idx, k=i)] = 1 / 4
        A[idx_to_A, np.diag(E_idx, k=i)] = 1 / 4
        A[idx_to_A, np.diag(S_idx, k=i+1)] = 1 / 4
        A[idx_to_A, np.diag(W_idx, k=i+1)] = 1 / 4
        A[idx_to_A, diag_terms[diag_index]] = n
        b.append(n)
        idx_to_A += 1
        diag_index += 1

    for i in range(-n, n):
        A[idx_to_A, np.diag(np.fliplr(N_idx), k=i)] = 1 / 4
        A[idx_to_A, np.diag(np.fliplr(W_idx), k=i)] = 1 / 4
        A[idx_to_A, np.diag(np.fliplr(S_idx), k=i+1)] = 1 / 4
        A[idx_to_A, np.diag(np.fliplr(E_idx), k=i+1)] = 1 / 4
        A[idx_to_A, anti_diag_terms[anti_diag_index]] = n
        b.append(n)
        idx_to_A += 1
        anti_diag_index += 1


    return A.tocsr(), np.array(b)

def construct_objective_fxn_derivs(n):
    n_squared = n**2
    weight_vector = np.hstack([
        (-n_squared / 4 / n_squared) * np.ones(4 * n_squared), # Other terms.
        (-n_squared / n)  * np.ones(4 * n), # Diagonals and anti-diagonals
    ])
    # We're cheating here and putting a factor into f that isn't in df and d2f
    # since they don't actually have to correspond.
    # It can help speed up convergence. Also, since this is the approximate
    # problem not the exact one, the values here don't matter.
    def f(x):
        return (weight_vector * (x * np.log(x))).sum() / n_squared - 3

    def df(x):
        return weight_vector * (np.log(x) + 1)

    def d2f(x):
        return weight_vector / x

    return f, df, d2f
