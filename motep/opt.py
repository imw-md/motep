from scipy.optimize import (
    OptimizeResult,
    differential_evolution,
    dual_annealing,
    minimize,
)


def callback(intermediate_result: OptimizeResult):
    """A callable called after each iteration."""
    print("Function value:", intermediate_result.fun)


def print_result(result: OptimizeResult):
    """Print `result`."""
    print("Optimization result:")
    print("  Message:", result.message)
    print("  Success:", result.success)
    print("  Status code:", result.status)
    print("  Number of function evaluations:", result.nfev)
    print("  Number of iterations:", result.nit)
    print("  Final parameters:", result.x)
    print("  Final function value:", result.fun)


def optimization_sa(fun, initial_guess, bounds):
    result = dual_annealing(
        fun,
        bounds=bounds,
        callback=callback,
        seed=40,
        x0=initial_guess,
    )
    print_result(result)
    return result.x


def optimization_nelder(fun, initial_guess, bounds):
    result = minimize(
        fun,
        initial_guess,
        bounds=bounds,
        method="Nelder-Mead",
        tol=1e-7,
        callback=callback,
        options={"maxiter": 100000},
    )
    print_result(result)
    return result.x


def optimization_bfgs(fun, initial_guess, bounds):
    result = minimize(
        fun,
        initial_guess,
        bounds=bounds,
        method="L-BFGS-B",
        tol=1e-7,
        callback=callback,
    )
    print_result(result)
    return result.x


def optimization_DE(fun, initial_guess, bounds):
    result = differential_evolution(
        fun,
        bounds,
        popsize=30,
        callback=callback,
    )
    print_result(result)
    return result.x
