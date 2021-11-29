#!/usr/bin/python3

import sys
import time

import numpy as np

from solvers.newton_diagonal import SeparableAffineCvxProblem
import problem_formulations.lower_bound as lower_bound

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    write_sol = '--write_sol' in sys.argv

    A, b = lower_bound.construct_A(n)
    f, df, d2f = lower_bound.construct_objective_fxn_derivs(n)

    initial_time = time.monotonic()
    x, v, norm_r, iters = SeparableAffineCvxProblem(A, b, f, df, d2f, 0).solve()
    final_time = time.monotonic()


    assert (x > 0).all() and (x < 1 / n).all()
    if write_sol:
        np.savez(f"{n}-lower-bound-solution", x=x, v=v)
    print(f"Reached a residual norm of {norm_r} in {iters} iterations in {final_time - initial_time} seconds")
    print(f"L_{n} = {lower_bound.calculate_lagrangian(A, b, n, v)}")
