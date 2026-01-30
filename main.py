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
from __future__ import annotations

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
]

# imports
import logging
import math
from decimal import Decimal
from numbers import Rational
from fractions import Fraction

logger = logging.getLogger(__name__)

class FormatError(Exception):
    """
    Exception raised when formatting does match to what is expected.
    """
    def __init__(self, msg="String formating is NOT ok"):
        super(FormatError, self).__init__(msg)
        self.msg = msg

def find_factors(n: int, pure=False) -> list[int]:
    """
    Returns all positive factors of an Integer.To get both negative and positive factors set pure=True
    :param n: The Integer whose factors are to be found.
    :param pure: Functions returns all factors, both -ve and +ve, if pure=True.
    :return: A sorted list of factors
    """
    n = abs(n)
    result = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            result.append(i)
            if n // i != i:
                result.append(n // i)

    if pure:
        result.extend([-i for i in result])
    return sorted(result)

def _validate(x: str) -> bool:
    if "|" not in x:
        return False
    if x[0] in "+-":
        x = x[1:]
    if not(x.startswith("[") and x.endswith("]")):
        return False
    return True

def _parse_rat(x: str) -> tuple[int, int]:
    try:
        p = int(x.strip())
        q = 1

    except ValueError:
        if not _validate(x):
            raise FormatError("Rat object shall be initialized in the format of Rat('[p|q]')")
        else:
            if x[0] in "+-":
                sign = -1 if x[0] == "-" else 1
                x = x[1:]
            else:
                sign = 1

            sep = x.index("|")
            p = sign * int(x[1:sep])
            q = int(x[sep + 1:-1])

    if q < 0:
        p = -p
        q = -q
    else:
        p = p
        q = q

    if q == 0:
        raise ZeroDivisionError

    high = math.gcd(p, q)
    p = p // high
    q = q // high

    return p, q

def _leading_unary(expr: str) -> tuple[int, str]:
    sign = 1
    i = 0

    while i < len(expr) and expr[i] in "+-":
        if expr[i] == "-":
            sign *= -1
        i += 1

    return sign, expr[i:]

class Rat(Rational):
    """An immutable Class for working with rational numbers.

    This class can be used for working with and manipulating rational numbers.
    To create a Rat object pass a string in the form of [p|q]. Notice that the middle separator is a vertical line.
    p represents the numerator and q represents the denominator.
    Objects of Rat can be used as numbers for comparison,assignment and arithmetic operators.

    Class Properties:
        - p
        - q
        - numerator
        - denominator
        - face
        - value

    Instance methods:
        - simplify
        - reciprocal
        - add
        - multiply
        - produce
        - eql
        - is_integer
        - as_tuple

    Static methods:
        - ftr
        - ftr_rep
        - frac_to_rat
        - rat_to_frac

    Class Methods:
        - from_float

    """

    __slots__ = ("_p", "_q", "_locked")

    def __init__(self,x: str | Rat | int, q: int | None = None) -> None:
        object.__setattr__(self, "_locked", False)

        if q is not None and isinstance(x, int):
            self._p, self._q = x, q
        elif q is not None and not isinstance(x, int):
            raise TypeError("p and q must be an integer")
        elif q is None and isinstance(x, float):
            tmp = Rat.ftr(x)
            self._p = tmp._p
            self._q = tmp._q
        elif q is None and isinstance(x, Rat):
            self._p, self._q = x._p, x._q
        else:
            self._p, self._q = _parse_rat(str(x))

        object.__setattr__(self, "_locked", True)

    @property
    def face(self) -> str:
        if self._q == 1:
            return f"{self._p}"
        return f"[{self._p}|{self._q}]"

    @property
    def value(self) -> float:
        return self._p / self._q

    @property
    def p(self) -> int:
        return self._p

    @property
    def q(self) -> int:
        return self._q

    @property
    def numerator(self) -> int:
        return self._p

    @property
    def denominator(self) -> int:
        return self._q

    @classmethod
    def from_float(cls, n: float) -> "Rat":
        return cls.ftr(n)

    @staticmethod
    def ftr(n: float | int | Decimal) -> Rat:
        """
        Converts a floating point number into a Rational number.
        :param n: a float
        :return: The equivalent Rational number in the form of a Rat object.
        """
        # getting number of decimal places
        n = float(n)
        num = Decimal(str(n))
        sign, digits, exp = num.as_tuple()
        if not isinstance(exp, int):
            raise ValueError("Decimal exponent is not an integer. NaN or Infinity are not supported")
        p = int("".join(map(str, digits)))
        q = 10 ** (-exp) if exp <= 0 else 1

        if sign:
            p = -p
        return Rat(p,q).simplify()

    @staticmethod
    def ftr_rep(b: float, a: float = 0) -> Rat:
        """
        Converts a repeating floating point number into a Rational number.
        :param b: the repeating part in the form of a float.For example 0.123123... b=0.123
        :param a: if the number starts to repeat after a certain digit 'a' is the float till that digit
        :return: The equivalent Rational number in the form of a Rat object.
        """
        if b <= 0 or b >= 1:
            raise ValueError("b is out of range.The repeating part must be between 0 and 1")
        a_dec = Decimal(str(a))
        b_dec = Decimal(str(b))

        # digits after decimal
        m = a_dec.as_tuple().exponent
        k = b_dec.as_tuple().exponent
        if not isinstance(m, int) or not isinstance(k, int):
            raise ValueError("Decimal exponent is not an integer. NaN or Infinity are not supported")

        m = -m
        k = -k

        n1 = Rat.ftr(a_dec)

        b_int = int(str(b_dec).replace(".", ""))
        rep_num = Rat(b_int,10 ** k - 1)

        shift = Rat(1|10 ** m)

        return (n1 + rep_num * shift).simplify()

    @staticmethod
    def frac_to_rat(x: Fraction):
        """
        Converts a Fraction object into a Rat object.
        :param x: A Fraction object.
        :return: The equivalent Rational number in the form of a Rat object.
        """
        return Rat(x.numerator, x.denominator)

    @staticmethod
    def rat_to_frac(x: Rat) -> Fraction:
        """
        Converts a Rat object into a Fraction object.
        :param x: A Rat object.
        :return: The equivalent fraction in the form of a Fraction object.
        """
        return Fraction(x.p, x.q)

    def simplify(self) -> Rat:
        """
        Simplifies rational numbers.
        :return: A simplified Rat object.
        """
        high = math.gcd(self._p, self._q)
        return Rat(self._p // high, self._q // high)

    def reciprocal(self) -> Rat:
        """
        :return: Reciprocal of self in form of a Rat object.
        """
        if self._p == 0:
            raise ZeroDivisionError
        return Rat(self._q, self._p).simplify()

    def add (self,other: Rat | int | float) -> Rat:
        """
        Adds two rational numbers.
        :param other: A Rat object.
        :return: The sum in form of a Rat object.
        """
        if isinstance(other, int):
            other = Rat(other)
        if isinstance(other, float):
            other = Rat.ftr(other)

        if self._q == other._q:
            return Rat(f"[{self._p + other._p}|{self._q}]")
        else:
            new_q = math.lcm(self._q, other._q)
            new_p = int(self._p * (new_q // self._q) + other._p * (new_q // other._q))
            return Rat(new_p,new_q)

    def multiply(self,other: Rat | int | float) -> Rat:
        """
        Multiplies two rational numbers.
        :param other: A Rat object.
        :return: The product in form of a Rat object.
        """
        if isinstance(other, int):
            other = Rat(other)
        if isinstance(other, float):
            other = Rat.ftr(other)

        return Rat(self._p * other._p, self._q * other._q)

    def produce(self,n:int) -> Rat:
        """
        Returns an equivalent rational number by multiplying n (The integer) to both numerator and denominator.
        :param n: an integer to be multiplied by the rational number.
        :return: The equivalent rational number in form of a Rat object.
        """
        return Rat(self._p * n, self._q * n)

    def _eql(self, other: Rat) -> tuple[Rat, Rat]:
        """
        Returns a tuple of the given rational number and self with same denominator.
        :param other: The other rational number.
        :return: A tuple of the two rational numbers
        """
        new_q = math.lcm(self._q, other._q)
        a = Rat(self._p * (new_q // self._q), new_q)
        b = Rat(other._p * (new_q // other._q), new_q)

        return a,b

    def is_integer(self) -> bool:
        """
        :return: True if the rational number is an integer, False otherwise.
        """
        return self._q == 1

    def as_tuple(self) -> tuple[int, int]:
        """
        returns a tuple of the rational number in form of a Rat object.
        :return: a tuple of the two rational numbers
        """
        return self._p, self._q

    def __add__(self, other: int | float | Rat) -> Rat:
        return self.add(other)

    def __radd__(self, other: int | float | Rat) -> Rat:
        return self.add(other)

    def __mul__(self, other: int | float | Rat) -> Rat:
        return self.multiply(other).simplify()

    def __rmul__(self, other: int | float | Rat) -> Rat:
        return self.multiply(other).simplify()

    def __neg__(self) -> Rat:
        return self.multiply(Rat(-1))

    def __pos__(self) -> Rat:
        return self

    def __sub__(self, other: int | float | Rat) -> Rat:
        return self.__add__(-other)

    def __rsub__(self, other: int | float | Rat) -> Rat:
        if isinstance(other, int):
            other = Rat(other)
        if isinstance(other, float):
            other = Rat.ftr(other)
        return other - self

    def __truediv__(self, other: int | float | Rat) -> Rat:
        if isinstance(other, int):
            other = Rat(other)
        if isinstance(other, float):
            other = Rat.ftr(other)
        return self.__mul__(other.reciprocal())

    def __rtruediv__(self, other: int | float | Rat) -> Rat:
        if isinstance(other, int):
            other = Rat(other)
        if isinstance(other, float):
            other = Rat.ftr(other)
        return self.reciprocal().multiply(other)


    def __floordiv__(self, other: int | float | Rat) -> int:
        return math.floor(self/other)

    def __rfloordiv__(self, other: int | float | Rat) -> int:
        return math.floor(other/self)

    def __mod__(self, other):
        raise NotImplementedError("Modulo not implemented for Rat")

    def __rmod__(self, other):
        raise NotImplementedError("Modulo not implemented for Rat")

    def __divmod__(self, other):
        raise NotImplementedError("Divmod not implemented for Rat")

    def __rdivmod__(self, other):
        raise NotImplementedError("Divmod not implemented for Rat")

    def __abs__(self) -> Rat:
        return Rat(f"[{abs(self._p)}|{abs(self._q)}]")

    def __pow__(self, power, modulo=None):
        if isinstance(power, int):
            if power < 0:
                return self.reciprocal() ** -power
            return Rat(int(self._p ** power),int(self._q ** power))
        else:
            return self.value ** power

    def __rpow__(self, other, modulo=None):
        return other ** self.value

    def __eq__(self, other) -> bool:
        if isinstance(other, Rat):
            a,b = self._eql(other)
            return a._p == b._p
        else:
            return self.value == other

    def __lt__(self, other) -> bool:
        if isinstance(other, Rat):
            tup = self._eql(other)
            return tup[0]._p < tup[1]._p
        else:
            return self.value < other

    def __le__(self, other) -> bool:
        if isinstance(other, Rat):
            tup = self._eql(other)
            return tup[0]._p <= tup[1]._p
        else:
            return self.value <= other

    def __gt__(self, other) -> bool:
        if isinstance(other, Rat):
            tup = self._eql(other)
            return tup[0]._p > tup[1]._p
        else:
            return self.value > other

    def __ge__(self, other) -> bool:
        if isinstance(other, Rat):
            tup = self._eql(other)
            return tup[0]._p >= tup[1]._p
        else:
            return self.value >= other

    def __ne__(self, other) -> bool:
        return not self == other

    def __getitem__(self, item) -> int:
        if isinstance(item, int):
            return (self._p, self._q)[item]
        else:
            if item.lower().strip() == "p":
                return self._p
            elif item.lower().strip() == "q":
                return self._q
            else:
                raise KeyError(f"Invalid key {item} for a Rat object.")

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False):
            raise AttributeError("Rat is immutable")
        super().__setattr__(name, value)

    def __round__(self, n=None):
        return round(self.value, n)

    def __floor__(self) -> int:
        return math.floor(self.value)

    def __float__(self) -> float:
        return self.value

    def __int__(self) -> int:
        return math.floor(self.value)

    def __trunc__(self):
        return int(self.value)

    def __ceil__(self):
        return math.ceil(self.value)

    def __format__(self, format_spec):
        return format(self.value, format_spec)

    def __str__(self) -> str:
        return self.face

    def __repr__(self) -> str:
        return f"Rat({self._p} | {self._q})"

    def __hash__(self):
        return hash((self._p, self._q))

    def __bool__(self):
        return self._p != 0

    def __copy__(self):
        return Rat(self._p, self._q)

    def __deepcopy__(self, memo):
        return Rat(self._p, self._q)

class Term:
    """Class for working with terms.
    This class can be used for working with and manipulating terms.
    To create an object of Term pass a string which includes a variable x.
    Coefficients can be int or Rat while powers of x can only be int.To represent exponentiation use '^' instead of '**'

    Class Properties:
        - face
        - degree

    Instance methods:
        - is_like
        - is_zero
        - multiply
        - evaluate

    Static methods:
        - group_multiply
        - shorten
        - arrange

    Class Methods:
        - from_parts
    """

    __slots__ = ("_power", "_coefficient", "_var")

    def __init__(self, x: str, var: str = "x"):
        var = var.lower().strip()
        if len(var) != 1 or not var.isalpha():
            raise FormatError("only alphabets are allowed as variables")
        self._var = var
        x = x.replace(var, "x")
        x = str(x).strip().lower()
        for i in x:
            if i in ["(",")","{","}"]:
                raise FormatError

        try:
            self._coefficient = Rat(x)
            self._power = 0
            return

        except FormatError:
            if x.count("x") != 1 or x.count("^") > 1:
                raise FormatError

            sign, x = _leading_unary(x)

            sep = x.index("x")
            if x[0] == "x":
                self._coefficient = Rat(1)
            else:
                try:
                    self._coefficient = Rat(x[:sep])
                except FormatError:
                    try:
                        self._coefficient = Rat.ftr(float(x[:sep]))
                    except ValueError:
                        raise FormatError("Could not parse the coefficient")

            self._coefficient = self._coefficient * sign

            variable_with_power = x[sep:]

            if "^" in variable_with_power:
                sep = variable_with_power.index("^")
                self._power = int(variable_with_power[sep + 1:])

            elif x[-1] != "x":
                raise FormatError

            else:
                self._power = 1

            if self._power < 0:
                raise ValueError("Negative powers are prohibited")

        self._normalize()

    @property
    def face(self) -> str:
        c = str(self._coefficient)
        if self._coefficient.q == 1:
            c = str(self._coefficient.p)
        if self._coefficient == 0:
            return "0"
        if self._power == 0:
            return f"{c}"
        if self._coefficient == 1:
            c = ""
        if self._power == 1:
            return f"{c}{self._var}"
        else:
            return f"{c}{self._var}^{self._power}"

    @property
    def var(self) -> str:
        return self._var

    @property
    def degree(self) -> int:
        return self._power

    @property
    def coefficient(self) -> Rat:
        return self._coefficient

    @coefficient.setter
    def coefficient(self, value: Rat):
        self._coefficient = value
        self._normalize()

    @property
    def power(self) -> int:
        return self._power

    @power.setter
    def power(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError("Power must be a positive integer")
        self._power = value
        self._normalize()

    @classmethod
    def from_parts(cls, coefficient: int | Rat, power: int, var: str = "x") -> Term:
        if not isinstance(coefficient, Rat):
            coefficient = Rat(coefficient)

        if not isinstance(power, int):
            raise TypeError("Power must be an int")

        if power < 0:
            raise ValueError("Negative powers are prohibited")

        var = var.lower().strip()
        if len(var) != 1 or not var.isalpha():
            raise FormatError("only alphabets are allowed as variables")

        # normalize zero
        if coefficient == 0:
            power = 0

        term = cls("1x")
        term._coefficient = coefficient
        term._power = power
        term._var = var
        return term

    @staticmethod
    def group_multiply(terms1: list[Term], terms2: list[Term]) -> list[Term]:
        """
        Multiplies two lists of terms.
        :param terms1: first list of terms
        :param terms2: second list of terms
        :return: a list of terms resulting from the multiplication
        """
        result = []
        for term1 in terms1:
            result.extend(term1.multiply(terms2))

        return Term.arrange(Term.shorten(result))

    @staticmethod
    def shorten(nums: list[Term]) -> list[Term]:
        """
            Simplifies a given list of terms by adding terms with same power.
            :param nums: a list of terms
            :return: a simplified list of terms with all terms having unique powers
            """
        powers: dict[int, Term] = {}

        for num in nums:
            if num._power not in powers:
                powers[num._power] = Term.from_parts(num._coefficient, num._power)
            else:
                powers[num._power] = powers[num._power] + num

        return list(powers.values())

    @staticmethod
    def arrange(terms: list[Term]) -> list[Term]:
        """
        Arranges a given list of terms in decreasing order of power.
        :param terms: a list of terms
        :return: a list of terms arranged in decreasing order of power
        """
        return sorted(terms, key=lambda term: term.power, reverse=True)

    def _normalize(self) -> None:
        if self._coefficient == 0:
            self._power = 0

    def is_constant(self) -> bool:
        return self._power == 0

    def is_like(self, other: Term) -> bool:
        """
        Returns True if both the terms are of same power
        :param self:  The first term to be compared
        :param other: The second term to be compared
        :return: True if both terms are of same power otherwise False
        """
        if not isinstance(other, Term):
            return False
        return self._power == other._power

    def multiply(self, others) -> list[Term]:
        """
        Used to multiply self with other term or a list of terms.
        :param others: a term or a list of terms
        :return: a list of terms resulting from the multiplication
        """
        if isinstance(others, (Term, Rat, int)):
            return [self * others]
        elif not isinstance(others, (list, tuple)):
            raise TypeError("only list of terms is accepted")
        return [self * other for other in others]

    def evaluate(self,n):
        """
        Evaluates value of a term at x=n
        :param n: value of x to evaluate. n can be Rat, int or float
        :return: the value of term at x=n
        """
        return self._coefficient*(n ** self._power)

    def is_zero(self) -> bool:
        return self._coefficient == 0

    def __add__(self, other: Term) -> Term:
        if not isinstance(other, Term):
            return NotImplemented
        if self._power == other._power:
            result = Term.from_parts(self._coefficient + other._coefficient, self._power,  var=self._var)
            result._normalize()
            return result
        else:
            raise ValueError("Cannot add terms with different powers")

    def __radd__(self, other) -> Term:
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, other: int | Rat | Term) -> Term:
        if isinstance(other,Term):
            return Term.from_parts(self._coefficient * other._coefficient, self._power + other._power, var=self._var)
        elif isinstance(other,Rat) or isinstance(other,int):
            return Term.from_parts(self._coefficient * other, self._power, var=self._var)
        else:
            raise TypeError(f"Cannot multiply a Term object with {type(other)}")

    def __rmul__(self, other: int | Rat | Term) -> Term:
        return self * other

    def __sub__(self, other) -> Term:
        if not isinstance(other, Term):
            return NotImplemented
        if self._power == other._power:
            result = Term.from_parts(self._coefficient - other._coefficient, self._power, var=self._var)
            result._normalize()
            return result
        else:
            raise ValueError("Cannot subtract terms with different powers")

    def __truediv__(self, other) -> Term:
        if isinstance(other, Term):
            if other.is_zero():
                raise ZeroDivisionError
            if other._power > self._power:
                raise ValueError("Operation will result in a negative power")
            return Term.from_parts(self._coefficient / other._coefficient, self._power - other._power, var=self._var)
        elif isinstance(other, int) or isinstance(other, Rat):
            if other == 0:
                raise ZeroDivisionError
            return Term.from_parts(self._coefficient / other, self._power, var=self._var)
        else:
            raise TypeError(f"Cannot divide a Term object with {type(other)}")

    def __eq__(self, other) -> bool:
        if not isinstance(other, Term):
            return False
        return self._power == other._power and self._coefficient == other._coefficient

    def __lt__(self, other) -> bool:
        if not isinstance(other, Term):
            return NotImplemented
        elif self._power == other._power:
            return self._coefficient < other._coefficient
        else:
            return self._power < other._power

    def __gt__(self, other) -> bool:
        if not isinstance(other, Term):
            return NotImplemented
        elif self._power == other._power:
            return self._coefficient > other._coefficient
        else:
            return self._power > other._power

    def __le__(self, other) -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other) -> bool:
        return self.__gt__(other) or self.__eq__(other)

    def __neg__(self) -> Term:
        return Term.from_parts(self._coefficient * -1, self._power, var=self._var)

    def __pow__(self, n: int) -> Term:
        if not isinstance(n, int) or n < 0:
            return NotImplemented
        if n == 0:
            return Term.from_parts(1, 0, var=self._var)
        if self.is_zero():
            return Term.from_parts(0, 0, var=self._var)
        return Term.from_parts(self._coefficient ** n, self._power * n, var=self._var)

    def __abs__(self) -> Term:
        return Term.from_parts(abs(self._coefficient), self._power, var=self._var)

    def __call__(self, n):
        return self.evaluate(n)

    def __bool__(self) -> bool:
        return self._coefficient != 0

    def __str__(self) -> str:
        return self.face

    def __repr__(self) -> str:
        return self.face

def unique(values, tolerance):
    result = []
    for v in values:
        if not any(abs(v - u) < tolerance for u in result):
            result.append(v)
    return result

class Polynomial:
    """
    This class is for working with mathematical polynomials.
    Note: Polynomials cannot have division of two expressions.

    Class Methods:
        - zero

    Class Properties:
        - face
        - data
        - degree

    Instance Methods:
        - add
        - multiply
        - subtract
        - derive
        - integrate
        - integer_zero
        - rational_zero
        - approximate_root
        - approximate_extrema
        - root_after
        - root_before
        - extrema_after
        - extrema_before
        - all_roots
        - all_extremes
        - area
        - evaluate_at
        - upward
        - downward

    Static Methods:
        - shorten
        - exp_eval
    """
    def __init__(self,expression: str | int | Rat, var: str = "x"):
        if len(var) != 1 or not var.isalpha():
            raise FormatError(msg="Variable must be a single alphabetic character")
        self.var = var.lower().strip()
        if isinstance(expression, int) or isinstance(expression, Rat) or str(expression).strip() == "0":
            self.terms = []
            return

        expression = expression.replace(self.var, "x")
        logger.debug("expression before cleaning: %s", expression)
        expression = expression.lower()
        allowed_symbols = ["+","-","*","^","(",")","[","]","|","0","1","2","3","4","5","6","7","8","9", "x"]
        expression = expression.replace(" ", "")
        if "x" not in expression:
            raise FormatError(msg="THERE HAS TO BE ONE VARIABLE!?!!")
        for i in expression:
            if i not in allowed_symbols:
                raise FormatError(msg="Formating of the given expression is not permitted in the grounds of the polynomials")


        expression = Polynomial.__clean(expression)
        terms = Polynomial.exp_eval(expression)
        self.terms = Term.arrange(Term.shorten(terms))

    @classmethod
    def zero(cls):
        obj = cls.__new__(cls)
        obj.terms = []
        obj.var = "x"
        return obj

    @property
    def face(self) -> str:
        if not self.terms:
            return "0"
        result = ""
        if self.terms[0].coefficient >= 0:
            result += f"{abs(self.terms[0])}"
        else:
            result += f"-{abs(self.terms[0])}"
        for term in self.terms[1:]:
            if term.coefficient >= 0:
                result += f" + {abs(term)}"
            else:
                result += f" - {abs(term)}"

        return result.replace("x", self.var)

    @property
    def data(self) -> dict[int, Rat]:
        return {x.power: x.coefficient for x in self.terms}

    @property
    def degree(self) -> int:
        if not self.terms:
            return -1
        return max(term.power for term in self.terms)


    @staticmethod
    def __clean(expr: str) -> str:
        """
        Cleans the expression by handling unary minus signs and removing redundant operators.
        :param expr: The expression string to be cleaned.
        :return: The cleaned expression string.
        """

        while any(x in expr for x in ("++", "--", "+-", "-+", "*+", "/+")):
            expr = expr.replace("++", "+")
            expr = str(expr) # ensure expr is str
            expr = expr.replace("--", "+")
            expr = expr.replace("+-", "-")
            expr = expr.replace("-+", "-")
            expr = expr.replace("*+", "*")
            expr = expr.replace("/+", "/")
        prev = ""
        point = len(expr)
        for i, sym in enumerate(expr):
            if prev + sym == "*-":
                depth = 0
                rat_depth = 0
                for j, ch in enumerate(expr[i + 1:]):
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                    elif ch == "[":
                        rat_depth += 1
                    elif ch == "]":
                        rat_depth -= 1
                    if depth == 0 and rat_depth == 0 and ch in "+-":
                        point = j + i + 1
                        break
                expr = expr[:i] + "(" + expr[i:point] + ")" + expr[point:]
            prev = sym
        while "(-(" in expr:
            expr = expr.replace("(-(", "(-1*(")
        if expr[0:2] == "-(":
            expr = "-1*(" + expr[2:]

        return expr

    @staticmethod
    def __implicate_multiplication(expr: str) -> str:
        """
        Inserts '*' where multiplication is implied, e.g.:
        (x+1)(x+2) -> (x+1)*(x+2)
        2(x+1)     -> 2*(x+1)
        x(x+1)     -> x*(x+1)
        """
        result = []
        prev = ""

        for ch in expr:
            if prev:
                # cases like ")(", "x(", "2(", ")x", ")2"
                if (
                        (prev.isdigit() or prev == "x" or prev == ")")
                        and (ch == "(" or ch == "x" or ch.isdigit()) and not (prev.isdigit() and ch.isdigit())
                ):
                    result.append("*")
            result.append(ch)
            prev = ch

        return "".join(result)

    @staticmethod
    def __single_layer_split(expr: str, ops: set[str]) -> list[str]:
        """
        Splits an expression at operators that are not inside any brackets.
        """
        parts = []
        rat_depth = 0
        depth = 0
        start = 0
        lead = ""

        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "[":
                rat_depth += 1
            elif ch == "]":
                rat_depth -= 1
            elif ch in ops and depth == 0 and rat_depth == 0:
                if expr[start:i] != "" and expr[start:i] not in ops:
                    parts.append(lead + expr[start:i])
                    lead = ""

                if ch == "-" and lead != "-":
                    lead = "-"
                elif ch == "-" and lead == "-":
                    lead = ""
                elif ch == "+" and lead == "-":
                    lead = "-"
                else:
                    lead = ""
                start = i+1

        if expr[start:] != "" and expr[start:] not in ops:
            parts.append(lead + expr[start:])
        return parts

    @staticmethod
    def __remove_outer_brackets(expr: str) -> str:
        if not expr:
            return expr

        expr = expr.strip()
        if not (expr.startswith("(") and expr.endswith(")")):
            return expr

        depth = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1

            # If depth hits 0 before the last char → outer brackets don't wrap all
            if depth == 0 and i != len(expr) - 1:
                return expr

        # Fully wrapped
        return expr[1:-1]

    @staticmethod
    def exp_eval(expr: str) -> list[Term]:
        expr = expr.strip()
        expr = Polynomial.__implicate_multiplication(expr)

        # remove outer brackets
        sign, expr = _leading_unary(expr)
        expr = Polynomial.__remove_outer_brackets(expr)

        # + / -
        parts = Polynomial.__single_layer_split(expr, {"+", "-"})
        if len(parts) > 1:
            if sign == -1:
                parts[0] = "-" + parts[0]
            terms = []
            for part in parts:
                terms += Polynomial.exp_eval(part)
            return Term.shorten(terms)

        # *
        factors = Polynomial.__single_layer_split(expr, {"*"})
        if len(factors) > 1:
            result = Polynomial.exp_eval(factors[0])
            for f in factors[1:]:
                result = Term.group_multiply(result, Polynomial.exp_eval(f))
                result = Term(f"{sign}x^0").multiply(result)
            return Term.shorten(result)

        # base case
        try:
            return [Term(expr)*sign]
        except FormatError:
            r = Rat(expr)
            return [Term(f"{r}x^0")*sign]

    def evaluate_at(self, n):
        """
        Returns f(x) at x=n.
        :param n: value of x to calculate value of polynomial.
        :return: f(n)
        """
        result = 0
        for i in self.terms:
            result += i.evaluate(n)

        return result

    def integer_zero(self) -> list[int]:
        """
        Finds integer zeros of the polynomial using Integer Root Theorem.If last constant is not an int, returns empty list.
        :return: An empty list if no integer zeros are found else a list of integer zeros.
        """
        self.terms = Term.arrange(self.terms)
        if not self.terms:
            return []

        num = self.data.get(0, 0)

        if not isinstance(num, int) and not num.is_integer():
            return []

        num = int(num)

        pos_zero = find_factors(num, pure=True)
        return [i for i in pos_zero if self.evaluate_at(i) == 0]

    def rational_zero(self) -> list[Rat]:
        """
        Finds rational zeros of the polynomial using Rational Root Theorem. If the coefficient of highest degree term
         or the constant term is not an int, returns empty list.
        :return: An empty list if no rational zeros are found else a list of rational zeros.
        """
        self.terms = Term.arrange(self.terms)
        if not self.terms:
            return []

        if not (self.terms[-1].is_constant() and not self.terms[-1].is_zero()):
            return []

        last = self.data.get(0, 0)
        first = self.data[self.terms[0].power]

        if not isinstance(last, int) or not isinstance(first, int):
            return []

        pos_zero = [Rat(p, q) for p in find_factors(last, pure=True) for q in find_factors(first, pure=True) if q != 0]
        return [i for i in pos_zero if self.evaluate_at(i) == 0]

    def derive(self) -> Polynomial:
        """
        Derives the polynomial using power rule.
        :return: The derivative of the polynomial in form of a Polynomial object.
        """
        derived_terms = []
        for term in self.terms:
            if term.power != 0:
                new_coefficient = term.coefficient * term.power
                new_power = term.power - 1
                derived_terms.append(Term.from_parts(new_coefficient, new_power, var=self.var))

        result_poly = Polynomial("0")
        result_poly.terms = Term.arrange(derived_terms)
        return result_poly

    def integrate(self) -> Polynomial:
        """
        Integrates the polynomial using power rule.
        ⚠️ NOTE ⚠️ => Constant of integration is not added.
        :return: The integral of the polynomial in form of a Polynomial object.
        """
        integrated_terms = []
        for term in self.terms:
            new_power = term.power + 1
            new_coefficient = term.coefficient / new_power
            integrated_terms.append(Term.from_parts(new_coefficient, new_power, var=self.var))

        result_poly = Polynomial("0")
        result_poly.terms = Term.arrange(integrated_terms)
        return result_poly

    def area(self, start=0, end=0):
        """
        Calculates the definite integral of the polynomial from start to end.
        :param start: The lower limit of integration.
        :param end: The upper limit of integration.
        :return: The area under the curve from start to end as a Rat object.
        """
        integral_poly = self.integrate()
        area = integral_poly.evaluate_at(end) - integral_poly.evaluate_at(start)
        return area

    @staticmethod
    def _same_sign(a, b) -> bool:
        if a == b:
            return True
        return (a > 0 and b > 0) or (a < 0 and b < 0)

    def approximate_root(self, start, end, tolerance=1e-7, max_iter=100) -> float:
        """
        Approximates a root of the polynomial in the given range using the Bisection Method.
        Assumes that the there is exactly one root in the given range.
        :param start: Initial point of the range.
        :param end: End point of the range.
        :param tolerance: The acceptable error margin for the approximation.
        :param max_iter: Maximum times that bisection will be done to find the root.
        :return: X value of the approximate root.
        """
        if Polynomial._same_sign(self.evaluate_at(start), self.evaluate_at(end)):
            raise ValueError("Function values at the interval endpoints must have opposite signs.")

        mid = (start + end) / 2
        f_mid = self.evaluate_at(mid)
        f_start = self.evaluate_at(start)
        iterations = 0
        while abs(f_mid) > tolerance and iterations < max_iter:
            if Polynomial._same_sign(f_start, f_mid):
                start = mid
                f_start = self.evaluate_at(start)
            else:
                end = mid
            mid = (start + end) / 2
            f_mid = self.evaluate_at(mid)
            iterations += 1

        return mid

    def approximate_extrema(self, start, end, tolerance=1e-7, max_iter=100) -> float:
        """
        Approximates an extremum of the polynomial in the given range using the Bisection Method on its derivative.
        Assumes that the there is exactly one extremum in the given range.

        :param start: Initial point of the range.
        :param end: End point of the range.
        :param tolerance: The acceptable error margin for the approximation.
        :param max_iter: Maximum times that bisection will be done to find the extrema.
        :return: X value of the approximate extremum.
        """
        derivative = self.derive()
        if Polynomial._same_sign(derivative.evaluate_at(start), derivative.evaluate_at(end)):
            raise ValueError("Function values at the interval endpoints must have opposite signs.")

        mid = (start + end) / 2
        f_mid = derivative.evaluate_at(mid)
        f_start = derivative.evaluate_at(start)
        iterations = 0
        while abs(f_mid) > tolerance and iterations < max_iter:
            if Polynomial._same_sign(f_start, f_mid):
                start = mid
                f_start = derivative.evaluate_at(start)
            else:
                end = mid
            mid = (start + end) / 2
            f_mid = derivative.evaluate_at(mid)
            iterations += 1

        return mid

    def upward(self, x) -> bool:
        """
        Determines if the polynomial is increasing after a given point.
        :param x: The point to evaluate.
        :return: True if the polynomial is increasing at x, False otherwise.
        """
        derivative = self.derive()
        return derivative.evaluate_at(x) > 0

    def downward(self, x) -> bool:
        """
        Determines if the polynomial is decreasing after a given point.
        :param x: The point to evaluate.
        :return: True if the polynomial is decreasing at x, False otherwise.
        """
        derivative = self.derive()
        return derivative.evaluate_at(x) < 0

    def root_after(self, start, tolerance=1e-7, max_iterations=32, initial_step = 1) -> float:
        """
        Finds the first root AFTER a given starting point by dynamically bracketing an interval.

        :param start: Point to start searching from.
        :param tolerance: Acceptable error for the root approximation.
        :param max_iterations: Maximum attempts to expand the search interval.
        :param initial_step: Initial step size to expand the search interval.
        :return: Approximated root as a float.
        :raises ValueError: If a root cannot be found within the maximum iterations.
        """
        iterations = 0
        f_start = self.evaluate_at(start)
        step = abs(initial_step)
        end = start + step
        f_end = self.evaluate_at(end)
        while Polynomial._same_sign(f_start, f_end):
            step *= 2
            end += step
            f_end = self.evaluate_at(end)
            iterations += 1
            if iterations > max_iterations:
                raise ValueError("Could not find a root in the range")

        return self.approximate_root(start, end, tolerance)

    def root_before(self, start, tolerance=1e-7, max_iterations=32, initial_step = 1) -> float:
        """
        Finds the first root BEFORE a given starting point by dynamically bracketing an interval.

        :param start: Point to start searching from.
        :param tolerance: Acceptable error for the root approximation.
        :param max_iterations: Maximum attempts to expand the search interval.
        :param initial_step: Initial step size to expand the search interval.
        :return: Approximated root as a float.
        :raises ValueError: If a root cannot be found within the maximum iterations.
        """
        iterations = 0
        f_start = self.evaluate_at(start)
        step = abs(initial_step)
        end = start - step
        f_end = self.evaluate_at(end)
        while Polynomial._same_sign(f_start, f_end):
            step *= 2
            end -= step
            f_end = self.evaluate_at(end)
            iterations += 1
            if iterations > max_iterations:
                raise ValueError("Could not find a root in the range")

        return self.approximate_root(start, end, tolerance)

    def extrema_after(self, start, tolerance=1e-7, max_iterations=32, initial_step = 1) -> float:
        """
        Finds the first extremum AFTER a given starting point by dynamically bracketing an interval.

        :param start: Point to start searching from.
        :param tolerance: Acceptable error for the extremum approximation.
        :param max_iterations: Maximum attempts to expand the search interval.
        :param initial_step: Initial step size to expand the search interval.
        :return: Approximated extremum as a float.
        :raises ValueError: If an extremum cannot be found within the maximum iterations.
        """
        derivative = self.derive()
        iterations = 0
        f_start = derivative.evaluate_at(start)
        step = abs(initial_step)
        end = start + step
        f_end = derivative.evaluate_at(end)
        while Polynomial._same_sign(f_start, f_end):
            step *= 2
            end += step
            f_end = derivative.evaluate_at(end)
            iterations += 1
            if iterations > max_iterations:
                raise ValueError("Could not find an extremum in the range")

        return self.approximate_extrema(start, end, tolerance)

    def extrema_before(self, start, tolerance=1e-7, max_iterations=32, initial_step = 1) -> float:
        """
        Finds the first extremum BEFORE a given starting point by dynamically bracketing an interval.

        :param start: Point to start searching from.
        :param tolerance: Acceptable error for the extremum approximation.
        :param max_iterations: Maximum attempts to expand the search interval.
        :param initial_step: Initial step size to expand the search interval.
        :return: Approximated extremum as a float.
        :raises ValueError: If an extremum cannot be found within the maximum iterations.
        """
        derivative = self.derive()
        iterations = 0
        f_start = derivative.evaluate_at(start)
        step = abs(initial_step)
        end = start - step
        f_end = derivative.evaluate_at(end)
        while Polynomial._same_sign(f_start, f_end):
            step *= 2
            end -= step
            f_end = derivative.evaluate_at(end)
            iterations += 1
            if iterations > max_iterations:
                raise ValueError("Could not find an extremum in the range")

        return self.approximate_extrema(end, start, tolerance)

    def root_range(self) -> tuple[float, float]:
        """
        Uses Cauchy's bound to find an interval in which all real roots of the polynomial lie.
        """
        a_n = abs(self.data[self.degree])

        if a_n == 0:
            raise ValueError("Leading coefficient cannot be zero")

        max_other = max(
            abs(c)
            for d, c in self.data.items()
            if d != self.degree
        )

        bound = 1 + max_other / a_n
        return -bound.value, bound.value

    def _linear_root(self):
        a = self.data.get(1, Rat(0))
        b = self.data.get(0, Rat(0))
        if a == 0:
            return []
        else:
            return [(-b/a).value]

    def all_roots(self, tolerance=1e-7):
        """
        Returns all approximated roots. If polynomial has integer or rational roots then those roots may be slightly
        in accurate due to float inaccuracies. Using integer_zero or rational_zero function is recommended.
        :param tolerance: Margin of error for roots.
        :return: List of roots
        """
        if self.degree == 1:
            return self._linear_root()
        derivative = self.derive()

        extremes = derivative.all_roots(tolerance=tolerance)
        extremes.sort()
        lower, upper = self.root_range()
        ranges = []
        prev = lower
        for i in extremes:
            ranges.append((prev, i))
            prev = i

        ranges.append((prev, upper))

        result = []
        for i in ranges:
            try:
                root = self.approximate_root(i[0], i[1], tolerance=tolerance)
                result.append(root)
            except ValueError:
                if abs(self.evaluate_at(i[0])) < tolerance:
                    result.append(i[0])
                pass

        if abs(self.evaluate_at(upper)) < tolerance:
            result.append(upper)

        result = unique(result, tolerance=tolerance)
        result.sort()

        return result

    def all_extremes(self, tolerance=1e-7):
        """
        Returns approximation of all extremes.
        :param tolerance: Margin of error for extremes.
        :return: A list of extremes
        """
        derivative = self.derive()
        return derivative.all_roots(tolerance=tolerance)

    def add(self,other: int | Rat | Term | Polynomial) -> Polynomial:
        """
        Adds two polynomials or a polynomial with a term or a rational number.
        :param other: A polynomial, term or rational number.
        :return: The sum in form of a Polynomial object.
        """
        if isinstance(other, Polynomial):
            new_terms = self.terms + other.terms
        elif isinstance(other, Term):
            new_terms = self.terms + [other]
        elif isinstance(other, Rat) or isinstance(other, int):
            new_terms = self.terms + [Term(f"{other}x^0")]
        else:
            raise TypeError("Unsupported type for addition with Polynomial")

        simplified_terms = Term.shorten(new_terms)
        result_poly = Polynomial("0")
        result_poly.terms = Term.arrange(simplified_terms)
        return result_poly

    def __add__(self, other: int | Rat | Term | Polynomial) -> Polynomial:
        return self.add(other)

    def __radd__(self, other: int | Rat | Term | Polynomial) -> Polynomial:
        return self.add(other)

    def multiply(self, other: int | Rat | Term | Polynomial) -> Polynomial:
        if isinstance(other, Polynomial):
            new_terms = Term.group_multiply(self.terms, other.terms)
            result_poly = Polynomial("0")
            result_poly.terms = Term.arrange(new_terms)
            return result_poly
        elif isinstance(other, Term):
            new_terms = Term.group_multiply(self.terms, [other])
            result_poly = Polynomial("0")
            result_poly.terms = Term.arrange(Term.shorten(new_terms))
            return result_poly
        elif isinstance(other, Rat) or isinstance(other, int):
            new_terms = Term.group_multiply(self.terms, [Term(f"{other}x^0")])
            result_poly = Polynomial("0")
            result_poly.terms = Term.arrange(Term.shorten(new_terms))
            return result_poly
        else:
            raise TypeError("Unsupported type for multiplication with Polynomial")

    def __mul__(self, other: int | Rat | Term | Polynomial) -> Polynomial:
        return self.multiply(other)

    def __rmul__(self, other: int | Rat | Term | Polynomial) -> Polynomial:
        return self.multiply(other)

    def __neg__(self) -> Polynomial:
        return self.multiply(-1)

    def subtract(self, other: int | Rat | Term | Polynomial) -> Polynomial:
        return self.add(-other)

    def __sub__(self, other: int | Rat | Term | Polynomial) -> Polynomial:
        return self.subtract(other)

    def __cal__(self, x):
        return self.evaluate_at(x)

    def __str__(self):
        return self.face
    def __repr__(self):
        return self.face

if __name__ == "__main__":
    print("This is a module for rational numbers, terms and polynomials.")
    print("Import this module to use its classes and methods.")