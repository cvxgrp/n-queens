# n-Queens Constant

This repo accompanies the paper [_Computing Tighter Bounds on the n-Queens Constant via Newton's Method_](https://web.stanford.edu/~boyd/papers/n_queens.html).

To get started with the code, clone this repo, and run `src/compute_upper_bound.py`
and `src/compute_lower_bound.py` to reproduce the results from the paper.
The code
requires that `scipy` and `numpy` are installed.
To verify the closed-form expressions used for the integrals, see
`src/problem_formulations/generate_closed_form_xqueenon_integral.sage`.
You will need [Sage Math](https://www.sagemath.org/) to generate the expressions.
