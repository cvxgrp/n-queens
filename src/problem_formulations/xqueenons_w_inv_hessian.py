
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

def vec_to_matrices(x, n):
    n_squared = n**2
    assert x.shape == (4 * n_squared + 8 * n -4,)
    N = x[:n_squared].reshape((n, n))
    E = x[n_squared:2 * n_squared].reshape((n, n))
    S = x[2 * n_squared:3 * n_squared].reshape((n, n))
    W = x[3 * n_squared:4 * n_squared].reshape((n, n))

    scaled_two_n_minus_SigNW = x[4 * n_squared: 4 * n_squared + 4 * n - 2:2]
    scaled_two_n_minus_SigSE = x[4 * n_squared + 1: 4 * n_squared + 4 * n - 2:2]
    scaled_two_n_minus_SigNE = x[4 * n_squared + 4 * n -2: 4 * n_squared + 8 * n:2]
    scaled_two_n_minus_SigSW = x[4 * n_squared + 4 * n + 1 -2: 4 * n_squared + 8 * n:2]
    return N, E, S, W, scaled_two_n_minus_SigNW, scaled_two_n_minus_SigSE, scaled_two_n_minus_SigNE, scaled_two_n_minus_SigSW

def construct_A(n):
    A = sp.dok_matrix((14 * n - 4 - 2, 4* n**2 + 8 * n - 4))
    x_index = np.arange(4 * n**2 + 8 * n -4)
    N_idx, E_idx, S_idx, W_idx, scaled_two_n_minus_SigNW, scaled_two_n_minus_SigSE, scaled_two_n_minus_SigNE, scaled_two_n_minus_SigSW = vec_to_matrices(x_index, n)

    idx_to_A = 0
    b = []

    # SigN_i = n
    for i in range(1, n):
        A[idx_to_A, N_idx[i,:]] = 1
        b.append(n)
        idx_to_A += 1

    # SigE_j = n
    for j in range(1, n):
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


    # SigNS_j = n
    for j in range(n):
        A[idx_to_A, N_idx[:,j]] = 1
        A[idx_to_A, S_idx[:,j]] = 1
        b.append(2 * n)
        idx_to_A += 1

    # SigEW_i = n
    for i in range(n):
        A[idx_to_A, E_idx[i,:]] = 1
        A[idx_to_A, W_idx[i,:]] = 1
        b.append(2 * n)
        idx_to_A += 1

    # SigNW_k <= 2n
    for k in range(1, 2 * n):
        A[idx_to_A, np.diag(np.fliplr(N_idx), n-k)] = 1
        A[idx_to_A, np.diag(np.fliplr(W_idx), n-k)] = 1
        A[idx_to_A, scaled_two_n_minus_SigNW[k - 1]] = 2 * n
        b.append(2 * n)
        idx_to_A += 1

    # SigSE_k <= 2n
    for k in range(1, 2 * n):
        A[idx_to_A, np.diag(np.fliplr(S_idx), n-k)] = 1
        A[idx_to_A, np.diag(np.fliplr(E_idx), n-k)] = 1
        A[idx_to_A, scaled_two_n_minus_SigSE[k - 1]] = 2 * n
        b.append(2 * n)
        idx_to_A += 1

    # SigNE_k <= 2n
    for k in range(1, 2 * n):
        A[idx_to_A, np.diag(N_idx, n-k)] = 1
        A[idx_to_A, np.diag(E_idx, n-k)] = 1
        A[idx_to_A, scaled_two_n_minus_SigNE[k - 1]] = 2 * n
        b.append(2 * n)
        idx_to_A += 1


    # SigSW_k <= 2n
    for k in range(1, 2 * n):
        A[idx_to_A, np.diag(S_idx, n-k)] = 1
        A[idx_to_A, np.diag(W_idx, n-k)] = 1
        A[idx_to_A, scaled_two_n_minus_SigSW[k - 1]] = 2 * n
        b.append(2 * n)
        idx_to_A += 1

    return A.tocsr(), np.array(b)

def construct_objective_fxn_derivs(n):
    four_n_squared = 4 * n**2
    single_entry_one_vec = np.array([1.])
    def g(x):
        return x * np.log(x)
    def dg(x):
        return np.log(x) + 1
    def d2g(x):
        return 1 / x
    log = np.log

    # Generatea functions (See generate_closed_form_xqueenon_integral.sage):
    def h(Sv, Nv):
        return 1/4*(2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)

    def dh_dS(Sv, Nv):
        return -Sv*log(Sv)/(Nv - Sv) + 1/4*(2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**2
    
    def dh_dN(Sv, Nv):
        return Nv*log(Nv)/(Nv - Sv) - 1/4*(2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**2

    def d2h_dS2(Sv, Nv):
        return  -log(Sv)/(Nv - Sv) - 2*Sv*log(Sv)/(Nv - Sv)**2 - 1/(Nv - Sv) + 1/2*(2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3

    def d2h_dN2(Sv, Nv):
        return  log(Nv)/(Nv - Sv) - 2*Nv*log(Nv)/(Nv - Sv)**2 + 1/(Nv - Sv) + 1/2*(2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3

    def hessian_upper_diag(Sv, Nv):
        return  2/(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3) - 2*(2*Nv*log(Nv)/(Nv - Sv)**2 + 2*Sv*log(Sv)/(Nv - Sv)**2 - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)**2/(((2*Nv*log(Nv)/(Nv - Sv)**2 + 2*Sv*log(Sv)/(Nv - Sv)**2 - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)**2/(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3) + 2*log(Sv)/(Nv - Sv) + 4*Sv*log(Sv)/(Nv - Sv)**2 + 2/(Nv - Sv) - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)*(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)**2)
    def hessian_lower_diag(Sv, Nv):
        return  -2/((2*Nv*log(Nv)/(Nv - Sv)**2 + 2*Sv*log(Sv)/(Nv - Sv)**2 - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)**2/(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3) + 2*log(Sv)/(Nv - Sv) + 4*Sv*log(Sv)/(Nv - Sv)**2 + 2/(Nv - Sv) - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)
    def hessian_off_diag(Sv, Nv):
        return  2*(2*Nv*log(Nv)/(Nv - Sv)**2 + 2*Sv*log(Sv)/(Nv - Sv)**2 - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)/(((2*Nv*log(Nv)/(Nv - Sv)**2 + 2*Sv*log(Sv)/(Nv - Sv)**2 - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)**2/(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3) + 2*log(Sv)/(Nv - Sv) + 4*Sv*log(Sv)/(Nv - Sv)**2 + 2/(Nv - Sv) - (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3)*(2*log(Nv)/(Nv - Sv) - 4*Nv*log(Nv)/(Nv - Sv)**2 + 2/(Nv - Sv) + (2*Nv**2*log(Nv) - 2*Sv**2*log(Sv) - Nv**2 + Sv**2)/(Nv - Sv)**3))


    def f(x):
        retval = -g(x[:four_n_squared]).sum() / (four_n_squared)
        _, _, _, _, NW, SE, NE, SW = vec_to_matrices(x, n)
        NW_ext = np.hstack([NW, single_entry_one_vec])
        SE_ext = np.hstack([single_entry_one_vec, SE])

        NE_ext = np.hstack([NE, single_entry_one_vec])
        SW_ext = np.hstack([single_entry_one_vec, SW])
        retval -= h(SE_ext, NW_ext).sum() / n
        retval -= h(SW_ext, NE_ext).sum() / n
        return -retval + 3

    def df(x):
        retval = np.empty_like(x)
        retval[:four_n_squared] = -dg(x[:four_n_squared]) / (four_n_squared)

        _, _, _, _, NW, SE, NE, SW = vec_to_matrices(x, n)
        NW_ext = np.hstack([NW, single_entry_one_vec])
        SE_ext = np.hstack([single_entry_one_vec, SE])

        NE_ext = np.hstack([NE, single_entry_one_vec])
        SW_ext = np.hstack([single_entry_one_vec, SW])
 

        # scaled_two_n_minus_SigNW = x[4 * n_squared: 4 * n_squared + 4 * n - 2:2]
        retval[four_n_squared: four_n_squared + 4 * n - 2:2] = -dh_dN(SE_ext, NW_ext)[:-1] / n
        # scaled_two_n_minus_SigSE = x[4 * n_squared + 1: 4 * n_squared + 4 * n - 2:2]
        retval[four_n_squared + 1: four_n_squared + 4 * n - 2:2] = -dh_dS(SE_ext, NW_ext)[1:] / n
        # scaled_two_n_minus_SigNE = x[4 * n_squared + 4 * n -2: 4 * n_squared + 8 * n:2]
        retval[four_n_squared + 4 * n - 2: four_n_squared + 8 * n:2] = -dh_dN(SW_ext, NE_ext)[:-1] / n
        # scaled_two_n_minus_SigSW = x[4 * n_squared + 4 * n + 1 -2: 4 * n_squared + 8 * n:2]
        retval[four_n_squared + 4 * n - 2 + 1: four_n_squared + 8 * n:2] = -dh_dS(SW_ext, NE_ext)[1:] / n
        return -retval

    def d2finv(x):
        diag = np.empty_like(x)
        diag[:four_n_squared] = -four_n_squared / d2g(x[:four_n_squared])

        _, _, _, _, NW, SE, NE, SW = vec_to_matrices(x, n)
        NW_ext = np.hstack([NW, single_entry_one_vec])
        SE_ext = np.hstack([single_entry_one_vec, SE])

        NE_ext = np.hstack([NE, single_entry_one_vec])
        SW_ext = np.hstack([single_entry_one_vec, SW])
        # scaled_two_n_minus_SigNW = x[4 * n_squared: 4 * n_squared + 4 * n - 2:2]
        diag[four_n_squared: four_n_squared + 4 * n - 2:2] = -hessian_upper_diag(SE_ext, NW_ext)[:-1] * n
        # scaled_two_n_minus_SigSE = x[4 * n_squared + 1: 4 * n_squared + 4 * n - 2:2]
        diag[four_n_squared + 1: four_n_squared + 4 * n - 2:2] = -hessian_lower_diag(SE_ext, NW_ext)[1:] * n
        # scaled_two_n_minus_SigNE = x[4 * n_squared + 4 * n -2: 4 * n_squared + 8 * n:2]
        diag[four_n_squared + 4 * n - 2: four_n_squared + 8 * n:2] = -hessian_upper_diag(SW_ext, NE_ext)[:-1] * n
        # scaled_two_n_minus_SigSW = x[4 * n_squared + 4 * n + 1 -2: 4 * n_squared + 8 * n:2]
        diag[four_n_squared + 4 * n - 2 + 1: four_n_squared + 8 * n:2] = -hessian_lower_diag(SW_ext, NE_ext)[1:] * n
        diag[4 * n**2 + 4 * n -3] = -n / d2h_dS2(SE_ext[-1], NW_ext[-1])
        diag[-1] = -n / d2h_dS2(SW_ext[-1], NE_ext[-1])


        offdiag = np.zeros(x.shape[0] - 1, dtype=x.dtype)
        offdiag[-4 * n +4::2] = -hessian_off_diag(SW_ext, NE_ext)[1:-1] * n
        offdiag[-8 * n +6:-4 * n+2:2] = -hessian_off_diag(SE_ext, NW_ext)[1:-1] * n

        return -sp.diags([offdiag, diag, offdiag], [-1, 0, 1])

    return f, df, d2finv

def construct_slack_augemented_vector_from_grid(N, E, S, W, A, b):
    grid_vec = np.hstack([N.reshape((-1)), E.reshape((-1)), S.reshape((-1)), W.reshape((-1)),])
    A_grid, A_slack = A[:, :grid_vec.shape[0]], A[:, grid_vec.shape[0]:]
    slack, istop = sp.linalg.lsmr(A_slack, b - A_grid @ grid_vec)[:2]
    assert istop <= 1
    return np.hstack([grid_vec, slack])
