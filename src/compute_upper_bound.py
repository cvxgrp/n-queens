
import time
import sys

import numpy as np

from solvers.newton_diagonal import SeparableAffineCvxProblem
from solvers.newton_sparse import SparseHessianAffineCvxProblem

import problem_formulations.approximate_xqueenons as approximate_xqueenons
import problem_formulations.xqueenons as xqueenons


if __name__ == '__main__':

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    write_sol = '--write_sol' in sys.argv

    # Approximate solver
    A, b = approximate_xqueenons.construct_A(n)
    f, df, d2f = approximate_xqueenons.construct_objective_fxn_derivs(n)
    
    initial_time = time.monotonic()
    x, v, norm_r, approx_iters = SeparableAffineCvxProblem(A, b, f, df, d2f, 0).solve()
    final_time = time.monotonic()
    approx_obj = f(x)
    approx_dur = final_time - initial_time

    # Exact form solver
    A, b = xqueenons.construct_A(n)
    f, df, d2f = xqueenons.construct_objective_fxn_derivs(n)

    N, E, S, W, _, _ = approximate_xqueenons.vec_to_matrices(x, n)
    x_init_sans_slack = np.hstack([N.reshape(-1), E.reshape(-1), S.reshape(-1), W.reshape(-1)])
    x_init_random = np.hstack([x_init_sans_slack, np.random.rand(A.shape[1] - x_init_sans_slack.shape[0])])
    x_init_deterministic = xqueenons.construct_slack_augemented_vector_from_grid(N, E, S, W, A, b)
    rand_p = .000001
    x_init = rand_p * x_init_random + (1 - rand_p) * x_init_deterministic


    initial_time = time.monotonic()
    x, v, norm_r, iters = SparseHessianAffineCvxProblem(A, b, f, df, d2f, 0, initial_point=x_init).solve()
    final_time = time.monotonic()

    obj = f(x)
    dur = final_time - initial_time

    if write_sol:
        np.savez(f"{n}-XQueenons-solution", x=x, v=v)

    print(f"Solved the approximate upper problem in {approx_dur} seconds and {approx_iters} iterations")
    print(f"Solved the upper problem in {dur} seconds and {iters} iterations with a residual norm of {norm_r}")
    print(f"U_{n} = {f(x)}")
