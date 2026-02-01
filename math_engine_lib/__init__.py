"""
Math_Engine

A lightweight symbolic mathematics engine for:
- Rational numbers (Rat)
- Algebraic terms (Term)
- Polynomials (Polynomial)

Features:
- Exact rational arithmetic
- Symbolic differentiation & integration
- Integer and rational root finding
- Numerical root and extrema approximation

Author: Lakshya Keswani
License: MIT
"""

# imports
from .utilities_file import FormatError, find_factors, unique
from .rat_file import Rat
from .term_file import Term
from .polynomial_file import Polynomial

# metadata
__title__ = "Math_Engine"
__version__ = "0.1.0"
__author__ = "Lakshya Keswani"
__license__ = "MIT"
__all__ = [
    "Rat",
    "Term",
    "Polynomial",
    "FormatError",
    "find_factors",
    "unique"
]


if __name__ == "__main__":
    print("This is a module for rational numbers, terms and polynomials.")
    print("Import this module to use its classes and methods.")