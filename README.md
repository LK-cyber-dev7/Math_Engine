# Math_Engine

Math_Engine is a Python-based mathematical engine designed to handle symbolic
and numeric computations. The project focuses on clean abstractions for
mathematical objects such as terms, polynomials, and expressions.

## Features
- Polynomial and term representation
- A Rational number class (`Rat`) with full numeric capabilities
- Simplification of expressions
- Modular and extensible design

## Rat Class

The `Rat` class is an immutable rational number class that fully behaves like
a Python numeric type. It can be used in arithmetic operations, comparisons,
and conversions while preserving exact rational values.

### Key Features
- **Immutable**: Once created, the numerator and denominator cannot be changed.
- **Arithmetic Operations**: Supports `+`, `-`, `*`, `/`, `**`, and reflected operations.
- **Comparisons**: Supports `<, <=, >, >=, ==, !=` with ints, floats, and other `Rat` objects.
- **Conversions**: Can convert to `float`, `int`, `round`, `floor`, `ceil`, and `trunc`.
- **Properties**: `.numerator`, `.denominator`, `.p`, `.q`, `.value`, `.face`.
- **Constructors**:
  - From string: `[p|q]`
  - From float: `Rat.ftr()`
  - From repeating decimals: `Rat.ftr_rep()`
- **Utility Methods**: `.simplify()`, `.reciprocal()`, `.produce(n)`, `.as_tuple()`, `.is_integer()`.
- **Interoperability**: Can convert to and from Python's `fractions.Fraction`.
- **Numbers ABC Compliance**: Inherits from `numbers.Rational` and works seamlessly with Python numeric libraries.

### Example Usage

```python
from math_engine import Rat
from fractions import Fraction

# Create Rat objects
r1 = Rat("[3|4]")
r2 = Rat(1, 2)
r3 = Rat.ftr(0.75)

# Arithmetic
sum_rat = r1 + r2
prod_rat = r1 * r2
pow_rat = r1 ** 2
rpow = 2 ** r1

# Comparison
print(r1 > r2)  # True

# Conversions
print(float(r1))        # 0.75
print(r1.numerator)     # 3
print(r1.denominator)   # 4

# Interoperability with Fraction
f = Fraction(3, 4)
r_from_f = Rat(f.numerator, f.denominator)

## Tech Stack
- Python 3
- Object-Oriented Programming

## Project Status
Actively under development 🚧

## Motivation
This project was created as a learning-focused initiative to improve
mathematical thinking, algorithm design, and Python programming skills.

## Author
Lakshya Keswani
