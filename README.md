# Math_Engine

Math_Engine is a Python-based mathematical engine designed to handle symbolic
and numeric computations. The project focuses on clean abstractions for
mathematical objects such as terms, polynomials, and expressions.

## Features
- Polynomial and term representation
- A Rational number class (`Rat`) with full numeric capabilities
- Counter class for flexible cyclic ranges and iterations.
- Simplification of expressions
- Modular and extensible design

## Counter Class

The Counter class provides a flexible way to cycle through a sequence of numbers or a custom dataset. It is immutable, meaning the underlying data cannot be modified once created, while still allowing iteration and pointer-based access.

### Features

- Endpoints inclusive for ranges.

- Supports int, float, and custom sequences (list or tuple).

- Floats are rounded to 12 decimal places to handle floating-point imperfections.

- Provides a pointer to track the current value, which can be moved forward, backward, or jumped to any index.

- Iteration or cycling does not change the main pointer.

### Class Methods
| Method                       | Description                                                    |
|------------------------------|----------------------------------------------------------------|
| `duplicate(cls, count_obj)`	 | Creates a copy of a Counter object. Useful for safe iteration. |
### Properties
| Property  | 	Description                   |
|-----------|--------------------------------|
| `value`   | 	Current value of the pointer. |
| `pointer` | 	Current pointer index.        |
### Instance Methods
| Method	     | Description                                             |
|-------------|---------------------------------------------------------|
| `advance()` | 	Move pointer forward (wraps to start if at the end).   |
| `back()	`   | Move pointer backward (wraps to end if at start).       |
| `jump(n)	`  | Jump pointer directly to index n.                       |
 | `peek(n) `  | Get value at index n without changing pointer.          |
 | `reset()	`  | Reset pointer to start.                                 |
 | `cycle()	`  | Generator to cycle through the dataset infinitely.      |
### Example Usage
```python
from math_engine_lib import Counter

# Range from 0 to 10 inclusive, step 2
c = Counter(10, 0, 2)  
print(c.value)       # 0
c.advance()
print(c.value)       # 2
print(c.peek(4))     # 8

# Cycling through values
cycler = c.cycle()
for _ in range(8):
    print(next(cycler), end=" ")  # 0 2 4 6 8 10 0 2

# Duplicate for safe iteration
c2 = Counter.duplicate(c)
for val in c2:
    print(val)
```

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
from math_engine_lib import Rat
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
```

## Term Class

The `Term` class represents a symbolic term of the form `coefficient * variable^power`.
It is designed for use in polynomials and algebraic expressions.

### Key Features
- **Components**: Each term has a coefficient (`Rat`) and a power (`int`).
- **Arithmetic Operations**: Supports addition, subtraction, multiplication, division (when powers allow), and exponentiation.
- **Comparisons**: Terms can be compared by power or coefficient.
- **Properties**: `.coefficient`, `.power`, `.var`.
- **Simplification**: Automatically simplifies terms with zero coefficients or powers of zero.
- **String Representation**: Displays in standard algebraic notation (e.g., `3/4*x^2`).
- **Interoperability**: Can be used directly in polynomials and combined with other `Term` objects.

### Example Usage
```python
from math_engine_lib import Term, Rat

# Create Term objects
t1 = Term("[3|4]x^2")  # 3/4 * x^2
t2 = Term("2a^2", var='a')        # 2 * a^2
t3 = Term.from_parts(Rat(1), 1, var='y')        # 1 * y

# Arithmetic
sum_term = t1 + t2               # Combines like terms
prod_term = t1 * t3              # Multiplies terms
pow_term = t1 ** 2                # Raises term to power

# Properties
print(t1.coefficient)            # Rat("[3|4]")
print(t1.power)                  # 2
print(t1.var)                    # 'x'

# Checks
print(t1.is_constant())            # False
print(t1.is_zero())                # False 

# String representation
print(t1)                        # '3/4*x^2'
```
## Polynomial Class

The Polynomial class represents a symbolic polynomial in one variable.
It supports exact algebraic manipulation as well as numerical approximation
for roots and extrema.

### Key Features

- Symbolic representation using Term objects

- Exact arithmetic via the Rat class

- Differentiation & integration

- Integer and rational root detection

- Numerical root & extrema approximation

- Automatic simplification and ordering

### Core Capabilities

- Addition, subtraction, multiplication

- Evaluation at numeric values

- Root finding (exact + approximate)

- Extrema detection via derivative analysis

- Definite area computation

### Example Usage
```python
from math_engine_lib import Polynomial, Rat

# Create a polynomial
p = Polynomial("x^3 - 3x + 1")

# Display
print(p)              # x^3 - 3x + 1
print(p.degree)       # 3

# Evaluation
print(p.evaluate_at(2))   # 3

# Derivative
dp = p.derive()
print(dp)             # 3x^2 - 3

# Roots (numerical)
roots = p.all_roots()
print(roots)

# Extremes
extremes = p.all_extremes()
print(extremes)

# Integration
ip = p.integrate()
print(ip)             # 1/4*x^4 - 3/2*x^2 + x

# Area under curve
area = p.area(0, 2)
print(area)
```

### Root Finding Strategy

- Cauchy’s Bound is used to locate a guaranteed root interval

- Derivative roots partition the domain into monotonic intervals

- Bisection Method is applied safely in each interval

- Duplicate roots are filtered using tolerance-based uniqueness

### This guarantees:

- No missed real roots

- Numerical stability

- Deterministic behavior

### Design Notes

- Polynomials are immutable in behavior (operations return new objects)

- All symbolic operations preserve exact rational arithmetic

- Numerical approximation is isolated and optional
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