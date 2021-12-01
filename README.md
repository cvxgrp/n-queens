# n-Queens Constant

This repo accompanies the paper [_Computing Tighter Bounds on the n-Queens Constant via Newton's Method_](https://web.stanford.edu/~boyd/papers/n_queens.html).

To get started with the code, clone this repo, and run `src/compute_upper_bound.py`
and `src/compute_lower_bound.py` to reproduce the results from the paper.
The code
requires that `scipy` and `numpy` are installed.
To verify the closed-form expressions used for the integrals, see
`src/problem_formulations/generate_closed_form_xqueenon_integral.sage`.
You will need [Sage Math](https://www.sagemath.org/) to generate the expressions.


We also provide witnesses for the bounds given in the paper, at `src/witnesses`.
For the upper bound problem, we provide the `x` for which `f(x) = U_1024`.
For the lower bound problem, we provide the `v` for which `L(v) = L_2048`.
To generate bounds from the witnesses run `src/upper_bound_from_witness.py` and
`src/lower_bound_from_witness.py`.

