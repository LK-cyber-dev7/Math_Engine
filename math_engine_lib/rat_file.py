from __future__ import annotations
import math
from decimal import Decimal
from numbers import Rational
from fractions import Fraction
from .utilities_file import FormatError

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