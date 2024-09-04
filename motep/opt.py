from scipy.optimize import (
    OptimizeResult,
    differential_evolution,
    dual_annealing,
    minimize,
)


def callback(intermediate_result: OptimizeResult):
    """A callable called after each iteration."""
    print("Function value:", intermediate_result.fun)


def optimization_sa(mytarget, initial_guess, bounds, *args):
    result = dual_annealing(
        mytarget,
        bounds=bounds,
        args=args,
        callback=callback,
        seed=40,
        x0=initial_guess,
    )
    print("Optimization result:")
    print("  Success:", result.success)
    print("  Message:", result.message)
    print("  Status code:", result.status)
    print("  Number of function evaluations:", result.nfev)
    print("  Final parameters:", result.x)
    print("  Final function value:", result.fun)
    return result.x


def optimization_nelder(mytarget, initial_guess, bounds, *args):
    result = minimize(
        mytarget,
        initial_guess,
        args=args,
        bounds=bounds,
        method="Nelder-Mead",
        tol=1e-7,
        callback=callback,
        options={"maxiter": 100000},
    )
    print("Optimization result:", result.message)
    print("Optimization result:")
    print("  Message:", result.message)
    print("  Success:", result.success)
    print("  Status code:", result.status)
    print("  Number of function evaluations:", result.nfev)
    print("  Number of iterations:", result.nit)
    print("  Final parameters:", result.x)
    print("  Final function value:", result.fun)
    return result.x


def optimization_bfgs(mytarget, initial_guess, bounds, *args):
    result = minimize(
        mytarget,
        initial_guess,
        args=args,
        bounds=bounds,
        method="L-BFGS-B",
        tol=1e-7,
        callback=callback,
    )
    print("Optimization result:", result.message)
    print("Optimization result:")
    print("  Message:", result.message)
    print("  Success:", result.success)
    print("  Status code:", result.status)
    print("  Number of function evaluations:", result.nfev)
    print("  Number of iterations:", result.nit)
    print("  Final parameters:", result.x)
    print("  Final function value:", result.fun)
    return result.x


def optimization_DE(mytarget, initial_guess, bounds, *args):
    result = differential_evolution(
        mytarget,
        bounds,
        args=args,
        popsize=30,
        callback=callback,
    )
    print("Optimization result:")
    print("  Message:", result.message)
    print("  Success:", result.success)
    print("  Number of function evaluations:", result.nfev)
    print("  Final parameters:", result.x)
    print("  Final function value:", result.fun)
    return result.x
