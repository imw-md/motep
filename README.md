# MOTEP

## Installation

```bash
pip install git+https://github.com/imw-md/motep.git
```

Optionally [our modified version of `mlippy`](https://gitlab.com/yuzie007/mlip-2/-/tree/mlippy) can be used.

As of 2024-09-20, three "engines" are implemented.

1. Mlippy based on python wrapper around mlip code
2. Numpy based calculator
3. Numba based calculator

## Usage

### `motep train`

```
motep train motep.toml
```

or

```
mpirun -np 4 motep train motep.toml
```

where the setting file `motep.toml` is like

```toml
data_training = 'training.cfg'
potential_initial = 'initial.mtp'
potential_final = 'final.mtp'

# seed = 10  # random seed for initializing MTP parameters

engine = 'numba'  # {'numpy', 'numba', 'mlippy'}

[loss]  # setting for the loss function
energy_weight = 1.0
forces_weight = 0.01
stress_weight = 0.001

# optimization steps

# style 1: simple
# steps = ['L-BFGS-B', 'Nelder-Mead']

# style 2: sophisticated
# "optimized" specifies which parameters are optimized at the step.
[[steps]]
method = 'L-BFGS-B'
optimized = ['scaling', 'species_coeffs', 'radial_coeffs', 'moment_coeffs']

[[steps]]
method = 'Nelder-Mead'
optimized = ['scaling', 'species_coeffs', 'radial_coeffs', 'moment_coeffs']
[steps.kwargs]
tol = 1e-7
[steps.kwargs.options]
maxiter = 1000
```

If some of the following parameters are already given in `initial.mtp`,
they are treated as the initial guess, which may or may not be optimized
depending on the above setting.

- `scaling`
- `radial_coeffs`
- `moment_coeffs`
- `species_coeffs`

### `motep grade`

```
motep grade motep.toml
```

```toml
data_training = 'traning.cfg'  # original data for training
data_in = 'in.cfg'  # data to be evaluated
data_out = 'out.cfg'  # data with `MV_grade`
potential_final = 'final.mtp'

seed = 42  # random seed
engine = 'numba'

# grade
algorithm = 'maxvol'  # {'maxvol', 'exaustive'}
```

### [`motep apply`](docs/apply.md)

### [`motep upconvert`](docs/upconvert.md)

## Big Question ?
What This code do

- code can read atomic data file in ASE atom and then calls write MTP file with random or custom unkown parameters and calculate energy,forces,stress
- write the function value (difference in weighted sum of energy,forces and stress between training set and current set based on current parameter)
- Uses python based optimization code to reduce the function value (errors)
- Can switch to different optimizer and any stage
- Can provide configurational weight to each atomic configurations
- Can optimize MTPs (You can optimize any force-field)
- New python based MTP calculator
- Switch to different radial basis type other then chebyshev such as Legendre/Cubic-spline/B-spline

````

Input Parameters ----- [Black/graybox]------Energy,forces,stress
   |                                              |
   |                                              |
   | -------------------|optimizer|---- Error compared to reference set


````

### Optimization algorithm

**Scipy modules :** ```L-BFGS-B```, ```Nelder-mead```, ```Differential evolution```, ```Dual Annealing```

### Additional Modules

This repository also contains a Python implementation of a [Genetic Algorithm (GA)](docs/ga.md) for optimization problems. The GA is designed to find the optimal solution to a given problem by evolving a population of candidate solutions over multiple generations.

## Authors

Pranav Kumar (trainer), Axel Forslund (calculator), Yuji Ikeda (testing)
