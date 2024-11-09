#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import scipy.sparse
import scipy.sparse.linalg
import numpy as np

dtype = np.longdouble

@dataclass
class InvHessianAffineCvxProblem:
    __doc__ = """
        A class to implement Newton-Raphson iteration on Knuth's Queenon problem. 

        A: a scipy.sparse.csr_matrix of shape (k, n). A may have any rank.
        b: a np.ndarray of shape (k,)
        f: This argument is ignored. It is not used by the algorithm.
        nabla_f: This function takes an input vector x and returns the gradient
            of f = sum_i f_i(x_i) wrt to x. It is R^n->R^n.
        lower_bound_f: If this is a float, all vectors passed to nabla_f and
            nabla_squared_f will be greater than it. If it is a (n,)
            vector, all vectors passed to nabla_f and nabla_squared_f
            will be elementwise greater than it. Defaults to -inf.
        upper_bound_f: If this is a float, all vectors passed to nabla_f and
            nabla_squared_f will be less than it. If it is a (n,)
            vector, all vectors passed to nabla_f and nabla_squared_f
            will be elementwise less than it. Defaults to +inf.
        initial_point: This is the initial point to start Newton iteration.
            It must be of shape (n,). This argument is optional.
        initial_duals: This is the initial value for the dual variables when
            we start Newton iteration. It must be of shape (k,). This argument
            is optional.
    """

    A: scipy.sparse.csr_matrix
    b: np.ndarray
    f: Optional[Callable[[np.ndarray], np.ndarray]]
    nabla_f: Callable[[np.ndarray], np.ndarray]
    inv_nabla_squared_f: Callable[[np.ndarray], scipy.sparse.dia_matrix]
    lower_bound_f: Union[float, np.ndarray] = field(default=float('-inf'))
    upper_bound_f: Union[float, np.ndarray] = field(default=float('inf'))
    initial_point: Optional[np.ndarray] = field(default=None)
    initial_duals: Optional[np.ndarray] = field(default=None)

    def solve(self, tolerance=1e-9, max_iters=50):
        """
            Returns:
                x: The optimal x, a shape (n,) np.ndarray.
                v: The optimal v, a shape (k,) np.ndarray.
                norm_r: The final residual norm of the iteration.
                iters: The number of iterations it took to find the solution.
        """
        return _newton_step(max_iters, tolerance, self)


def _newton_step(max_iters, tolerance, problem):
    dim_constraint, dim_variables  = problem.A.shape
    A = problem.A
    b = problem.b
    nabla_f = problem.nabla_f
    inv_nabla_squared_f = problem.inv_nabla_squared_f
    upper_bound_f = problem.upper_bound_f
    if np.asarray(upper_bound_f).shape == ():
        upper_bound_f = upper_bound_f * np.ones(dim_variables)

    lower_bound_f = problem.lower_bound_f
    if np.asarray(lower_bound_f).shape == ():
        lower_bound_f = lower_bound_f * np.ones(dim_variables)

    def x_is_valid(x):
        return (x > lower_bound_f).all() and (x < upper_bound_f).all()

    class PrimalDualPair:
        """
            This class is a way to cache frequently accessed values like
            the residuals associated with a given x, v pair and their norms.

            x MUST be greater than lower_bound_f and less than upper_bound_f.
        """
        x: np.ndarray
        v: np.ndarray

        def __init__(self, x: np.ndarray, v: np.ndarray):
            self.x = x
            self.v = v
            # When this code is well-tested, delete the assert
            assert x_is_valid(x), "PrimalDualPair is illegaly constructed." \
                " This is a bug in the solver, not in your code."
            self.r_pri = A @ x - b
            self.r_dual = nabla_f(x) + A.T @ v

            self.norm_r_pri = np.linalg.norm(self.r_pri)
            self.norm_r_dual = np.linalg.norm(self.r_dual)
            self.norm_r = np.sqrt(self.norm_r_pri**2 + self.norm_r_dual**2)

    def build_minres_LHS(inv_nabla_squared_f_x):
        def matvec(v):
            z1 = A.T @ v
            z2 = inv_nabla_squared_f_x @ z1
            z3 = A @ z2
            return z3

        return scipy.sparse.linalg.LinearOperator(shape=(A.shape[0], A.shape[0]),
                                                  matvec=matvec)


    def compute_update_direction(pair):
        """
            Here we compute the update direction of a given (x, v) pair.
            The algorithm was derived on a whiteboard in Packard 243.

            It comes from solving the optimality conditions:
                nabla f(x) + nabla^2 f(x) dx + A^T (v + dv) = 0
                A (x + dx) - b = 0
            The final solution is:
                dv = -(A @ nabla^2f(x)^-1 @ A^T)^-1 @ (-r_p + A @ nabla^2f(x)^-1 @ r_d)
                dx = -nabla^2f(x)^-1 @ (r_d + A^T @ dv)

            Note that we exploit the seperability of f here in how we invert
            nabla^2f(x).
        """
        inv_nabla_squared_f_x = inv_nabla_squared_f(pair.x).tocsc()

        inv_nabla_squared_f_r_dual = inv_nabla_squared_f_x @ pair.r_dual
        #LHS = A @ scipy.sparse.linalg.spsolve(nabla_squared_f_x, A.T)
        LHS = build_minres_LHS(inv_nabla_squared_f_x)
        RHS = -pair.r_pri + A @ inv_nabla_squared_f_r_dual
        neg_dv, stop_reason = scipy.sparse.linalg.minres(
                LHS, RHS
        )

        assert stop_reason == 0, "minres should conclude that it succeeded"
        dv = -neg_dv
        dx = -(inv_nabla_squared_f_x @ (pair.r_dual + A.T @ dv))
        return dx, dv

    def max_t_for_valid_step(x, dx):
        """
        To try and speed up convergence speed for the Newton step, we use a
        strategy that ensures the initial step is valid. In theory, we could
        use this guarantee to eliminate the feasibility tests.

        We find t_max with sup{t >= 0 | for all i: x_i + t dx_i in [l_i, u_i]}

        If dx_i > 0, we have to worry about the upper bound. Otherwise, we don't.
            This entails computing x_i + t_i dx_i = u_i => t_i = (u_i - x_i) / dx_i.
        If dx_i < 0, we have to worry about the lower bound. Otherwise, we don't.
            This entails computing x_i + t_i dx_i = l_i => t_i = (l_i - x_i) / dx_i.
        If dx_i == 0, we can let t_i = 1.

        The final solution is therefore min({t_i for all i})

        Note: If t_max > 1, we return the value greater than 1 in order to make
            it easier to process the output later.
        """
        upper_mask = (dx > 0.0) & np.isfinite(upper_bound_f)
        upper_min_t = (upper_bound_f[upper_mask] - x[upper_mask]) / dx[upper_mask]
        
        lower_mask = (dx < 0.0) & np.isfinite(lower_bound_f)
        lower_min_t = (lower_bound_f[lower_mask] - x[lower_mask]) / dx[lower_mask]

        t_max = min(lower_min_t.min(initial=np.inf), upper_min_t.min(initial=np.inf))

        return t_max

    def infeasible_Newton_step(oldpt, dx, dv):
        """
            Here we implement the infeasible start Newton algorithm from Section
            10.3.2 of Convex Optimization by Boyd and Vandenberghe.

            We start with a t that ensures we're below (above) the lower (upper)
            bound.

            Neither alpha nor beta have been carefully tuned.
        """
        max_t = max_t_for_valid_step(oldpt.x, dx)
        t = min(1, 0.95 * max_t)
        beta = 0.9
        alpha = 0.01
        newpt = PrimalDualPair(oldpt.x + t * dx, oldpt.v + t * dv)

        while (1 - alpha * t) * oldpt.norm_r < newpt.norm_r:
            t *= beta
            newpt = PrimalDualPair(oldpt.x + t * dx, oldpt.v + t * dv)

        return newpt, t

    def construct_x_from_bounds(lower_bound_f, upper_bound_f):
        initial_x = np.nan * np.ones(dim_variables, dtype=dtype)

        both_infinite_mask = np.isinf(lower_bound_f) & np.isinf(upper_bound_f)
        initial_x[both_infinite_mask] = 0.0
        del both_infinite_mask
        lower_infinite_mask = np.isinf(lower_bound_f) & np.isfinite(upper_bound_f)
        initial_x[lower_infinite_mask] = upper_bound_f[lower_infinite_mask] - 1.0
        del lower_infinite_mask
        upper_infinite_mask = np.isfinite(lower_bound_f) & np.isinf(upper_bound_f)
        initial_x[upper_infinite_mask] = lower_bound_f[upper_infinite_mask] + 1.0
        del upper_infinite_mask
        both_finite_mask = np.isfinite(lower_bound_f) & np.isfinite(upper_bound_f)
        initial_x[both_finite_mask] = (lower_bound_f[both_finite_mask] + upper_bound_f[both_finite_mask]) / 2
        del both_finite_mask
        assert not np.isnan(initial_x).any() and not np.isinf(initial_x).any(), "Initial x constructed incorrectly"
        return initial_x

    if problem.initial_point is not None:
        initial_x = problem.initial_point
        assert x_is_valid(initial_x), "Provided x is not feasible. This is a bug" \
            " in your provided x, not the solver."
    else:
        initial_x = construct_x_from_bounds(lower_bound_f, upper_bound_f)
        assert x_is_valid(initial_x), "Constructed x is not feasible." \
            " This is a bug in the solver not your code."

    if problem.initial_duals is not None:
        initial_v = problem.initial_duals
    else:
        # Solve nabla_f(x0) + A^T v = 0 to find initial_v
        initial_v, istop_reason = scipy.sparse.linalg.lsmr(
            A.T, -nabla_f(initial_x)
        )[:2]
        assert istop_reason <= 2, "Honestly, not sure what's reasonable to expect here."

    pd_pair = PrimalDualPair(initial_x, initial_v)

    iters = 0

    while np.abs(pd_pair.norm_r) > tolerance and iters < max_iters:
        dx, dv = compute_update_direction(pd_pair)
        pd_pair, _ = infeasible_Newton_step(pd_pair, dx, dv)
        iters += 1

    return pd_pair.x, pd_pair.v, pd_pair.norm_r, iters
