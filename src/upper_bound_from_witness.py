import sys

import numpy as np

import problem_formulations.xqueenons as xqueenons


file_name = sys.argv[1] if len(sys.argv) > 1 else 'witnesses/U_1024.npz'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
x = np.load(file_name)['x']

f, _, _ = xqueenons.construct_objective_fxn_derivs(n)

print(f"U_{n} = {f(x)}")
