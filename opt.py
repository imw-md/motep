from scipy.optimize import differential_evolution, dual_annealing, minimize


def optimization_sa(mytarget, initial_guess, bounds, *args):
    def callback_print(params, f, context):
        func_value = mytarget(params, *args)
        print("Function value:", func_value)
        # print("Parameters:", params)

    result = dual_annealing(
        mytarget,
        bounds=bounds,
        args=args,
        callback=callback_print,
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
    def callback_print(params):
        func_value = mytarget(params, *args)
        print("Function value:", func_value)
        # print("Parameters:", params)

    result = minimize(
        mytarget,
        initial_guess,
        args=args,
        bounds=bounds,
        method="Nelder-Mead",
        tol=1e-7,
        callback=callback_print,
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
    def callback_print(params):
        func_value = mytarget(params, *args)
        print("Function value:", func_value)
        # print("Parameters:", params)

    result = minimize(
        mytarget,
        initial_guess,
        args=args,
        bounds=bounds,
        method="L-BFGS-B",
        tol=1e-7,
        callback=callback_print,
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
    def callback_print(xk, convergence):
        print("Objective function values:", mytarget(xk, *args))
        # You can add more details about the optimization process here

    result = differential_evolution(
        mytarget, bounds, args=args, popsize=30, callback=callback_print
    )

    print("Optimization result:")
    print("  Message:", result.message)
    print("  Success:", result.success)
    print("  Number of function evaluations:", result.nfev)
    print("  Final parameters:", result.x)
    print("  Final function value:", result.fun)
    return result.x
