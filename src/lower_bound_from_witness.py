import sys

import numpy as np

import problem_formulations.lower_bound as lower_bound


file_name = sys.argv[1] if len(sys.argv) > 1 else 'witnesses/L_2048.npz'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
v = np.load(file_name)['v']

A, b = lower_bound.construct_A(n)


print(f"L_{n} = {lower_bound.calculate_lagrangian(A, b, n, v)}")
