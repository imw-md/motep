from ikdtools.io.mlip.cfg import write_cfg
from ikdtools.io.mlip.cfg import read_cfg
import numpy as np
from ase.calculators.morse import MorsePotential
import copy
from scipy.optimize import minimize
import mlippy

from scipy.optimize import differential_evolution
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as minimize_GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from mpi4py import MPI

from scipy.optimize import dual_annealing

def optimization_sa(mytarget, initial_guess, *args):
    bounds = [(-1e6, 1e6)] + [(-10, 10)] * (len(initial_guess) - 1)
    
    def callback_print(params,f,context):
        func_value = mytarget(params, *args)
        print("Function value:", func_value)
        #print("Parameters:", params)

    result = dual_annealing(mytarget, bounds=bounds, args=args,callback=callback_print,seed=40,x0=initial_guess)
    
    print("Optimization result:")
    print("  Success:", result.success)
    print("  Message:", result.message)
    print("  Status code:", result.status)
    print("  Number of function evaluations:", result.nfev)
    print("  Final parameters:", result.x)
    print("  Final function value:", result.fun)
    
    return result.x



    




def optimization_nelder(mytarget,initial_guess,*args):
    bounds = [(-1e6, 1e6)] + [(-10, 10)] * (len(initial_guess) - 1)
    def callback_print(params):
        func_value = mytarget(params, *args)
        print("Function value:", func_value)
        #print("Parameters:", params)

    result = minimize(mytarget, initial_guess, args=args, bounds=bounds, method="Nelder-Mead",tol=1e-7,callback=callback_print,options={'maxiter': 100000})
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


def optimization_bfgs(mytarget,initial_guess,*args):
    bounds = [(-1e6, 1e6)] + [(-0.1, 0.1)] * (len(initial_guess) - 1)
    def callback_print(params):
        func_value = mytarget(params, *args)
        print("Function value:", func_value)
        #print("Parameters:", params)

    result = minimize(mytarget, initial_guess, args=args, bounds=bounds, method="L-BFGS-B",tol=1e-7,callback=callback_print)
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





def optimization_DE(mytarget,initial_guess, *args):
    bounds = [(-1e6, 1e6)] + [(-0.1, 0.1)] * (len(initial_guess) - 1)

    def callback_print(xk, convergence):
        print("Objective function values:", mytarget(xk, *args))
        # You can add more details about the optimization process here

    result = differential_evolution(mytarget, bounds, args=args, popsize=30,callback=callback_print)

    print("Optimization result:")
    print("  Message:", result.message)
    print("  Success:", result.success)
    print("  Number of function evaluations:", result.nfev)
    print("  Final parameters:", result.x)
    print("  Final function value:", result.fun)
    return result.x



class MyProblem(ElementwiseProblem):
    def __init__(self, mytarget, initial_guess,*args):
        self.args = args
        self.initial_guess= initial_guess  # initial guess is only for determining unknowns
        self.mytarget = mytarget
        super().__init__(n_var=len(self.initial_guess),
                         n_obj=1,
                         xl=np.array([-1000]+[-1]*(len(initial_guess)-1)),
                         xu=np.array([-1000]+[1]*(len(initial_guess)-1)))
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"]=self.mytarget(x, *self.args)

def optimization_pymoo(mytarget,initial_guess, *args):
    problem = MyProblem(mytarget,initial_guess,*args)
    algorithm= GA(pop_size=50,eliminate_duplicates=True)
    termination = get_termination("n_gen", 30)
    res = minimize_GA(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)
    best_solution =res.X
    return best_solution
