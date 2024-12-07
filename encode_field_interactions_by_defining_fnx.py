#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, exp, Function, simplify

# Define symbolic variables
x = symbols('x')  # Spatial variable
n = symbols('n', integer=True)  # Recursive layer index
alpha = symbols('alpha')  # Scaling factor
beta = symbols('beta')  # Damping factor

# Define the recursive function f_n(x)
def define_f_n():
    """
    Defines the recursive function f_n(x) rigorously.
    Returns a symbolic expression for f_n(x).
    """
    f = Function('f')  # Recursive function placeholder
    # Base case: f_0(x) = e^(-x^2) (Gaussian distribution for locality)
    f_0 = exp(-x**2)

    # Recursive case: f_n(x) = alpha^n * f_0(x) * exp(-beta * n)
    f_n = alpha**n * f_0 * exp(-beta * n)
    
    return simplify(f_n)

# Generate and display f_n(x)
f_n_expression = define_f_n()
print(f"Recursive Function f_n(x): {f_n_expression}")

# Plotting f_n(x) for visual beauty and understanding
def plot_f_n(alpha_val=0.5, beta_val=0.1, layers=5):
    """
    Plots the recursive function f_n(x) for multiple layers.
    """
    x_vals = np.linspace(-3, 3, 500)
    f_0_vals = np.exp(-x_vals**2)
    
    plt.figure(figsize=(10, 6))
    for n in range(layers):
        scaling = (alpha_val**n) * np.exp(-beta_val * n)
        f_n_vals = scaling * f_0_vals
        plt.plot(x_vals, f_n_vals, label=f"Layer n={n}")
    
    plt.title("Visualization of Recursive Function $f_n(x)$")
    plt.xlabel("$x$")
    plt.ylabel("$f_n(x)$")
    plt.legend()
    plt.grid()
    plt.show()

plot_f_n()
