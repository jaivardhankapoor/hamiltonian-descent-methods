# ee609
Hamiltonian Descent Methods implementation

## Components

### optimizers.py

We have implemented from scratch, 6 optimizer classes:
 1. GradientDescent
 2. Momentum
 3. ExplicitMethod1
 4. ExplicitMethod2
 5. StochasticExplicitMethod1
 6. StochasticExplicitMethod1

 The inputs they require are starting points, hyperparameters, and function class, which contains functions and gradients of the objective, noise and kinetic map.

### objectives.py

We have implemented 4 functions here:
1. Psi (Used in 2nd Explicit Method)
2. PowerFunction2D (Uses coefficients for skewing, and degree of function required)
3. PowerFunctionShifted (returns a shifted version of the Power function with 0 skew)
4. Noise2D (returns noise for gradient)

Each function requires its own parameters. The class methods are f(), k(), and their respective gradients. Gradients are computed using the autograd library in Python.

### plots.py

We put here all plotting routines that we use to generate the figures in the paper.

## Notes on novelty of code:

All code presented in this folder is fully handwritten, except some snippets in plots.py that we found from matplotlib documentation. Code is modlar enough for usage in real-world tasks, and can be easily extended for other objectives and optimizers.
