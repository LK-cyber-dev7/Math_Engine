"""
This module can be used for working with rational numbers, terms, and polynomials.
"""
# imports
from __future__ import annotations
import logging
import math
from decimal import Decimal
from numbers import Rational
from fractions import Fraction

logging.basicConfig(level=logging.DEBUG)

class FormatError(Exception):
    """
    Exception raised when formatting does match to what is expected.
    """
    def __init__(self, msg="String formating is NOT ok"):
        super(FormatError, self).__init__(msg)
        self.msg = msg

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
        else:
            self._p, self._q = _parse_rat(str(x))

        object.__setattr__(self, "_locked", True)

    @property
    def face(self) -> str:
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
            return Rat(f"[{int(self._p ** power)}|{int(self._q ** power)}]")
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

    Instance methods:
        - is_equal
        - simplify
        - multiply
        - evaluate

    Static methods:
        - group_multiply
        - arrange
    """

    def __init__(self, x: str):
        for i in x:
            if i in ["(",")","{","}"]:
                raise FormatError

        if x.count("x") != 1:
            raise FormatError

        sign, x = _leading_unary(x)

        sep = x.index("x")
        if x[0] == "x":
            self.coefficient = Rat(1)
        else:
            try:
                self.coefficient = Rat(x[:sep])
            except FormatError:
                try:
                    self.coefficient = Rat.ftr(float(x[:sep]))
                except ValueError:
                    raise FormatError("Could not parse the coefficient")

        self.coefficient = self.coefficient * sign

        self.variable_with_power = x[sep:]

        if "^" in self.variable_with_power:
            sep = self.variable_with_power.index("^")
            self.power = int(self.variable_with_power[sep + 1:])

        elif x[-1] != "x":
            raise FormatError

        else:
            self.power = 1



    @property
    def face(self) -> str:
        c = str(self.coefficient)
        if self.coefficient.q == 1:
            c = str(self.coefficient.p)
        if self.power == 0:
            return f"{c}"
        if self.coefficient.p == 1:
            c = ""
        if self.power == 1:
            return f"{c}x"
        else:
            return f"{c}x^{self.power}"

    def is_equal(self, other: Term) -> bool:
        """
        Returns True if both the terms are of same power
        :param self:  The first term to be compared
        :param other: The second term to be compared
        :return: True if both terms are of same power otherwise False
        """
        return self.power == other.power

    def __add__(self, other: Term) -> Term:
        if self.power == other.power:
            return Term(f"{self.coefficient + other.coefficient}x^{self.power}")
        else:
            raise ValueError("Cannot add terms with different powers")

    def __radd__(self, other: Term) -> Term:
        return self + other

    def __mul__(self, other: int | Rat | Term) -> Term:
        if isinstance(other,Term):
            return Term(f"{self.coefficient * other.coefficient}x^{self.power+other.power}")
        elif isinstance(other,Rat) or isinstance(other,int):
            return Term(f"{self.coefficient * other}x^{self.power}")
        else:
            raise TypeError(f"Cannot multiply a Term object with {type(other)}")

    def __rmul__(self, other: int | Rat | Term) -> Term:
        return self * other

    def multiply(self, others) -> list[Term]:
        """
        Used to multiply self with other term or a list of terms.
        :param others: a term or a list of terms
        :return: a list of terms resulting from the multiplication
        """
        if isinstance(others,Term) or isinstance(others,Rat):
            return [self * others]
        else:
            result = []
            for other in others:
                result.append(self * other)
            return result

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

        return Term.arrange(Polynomial.shorten(result))

    def __sub__(self, other) -> Term:
        if self.power == other.power:
            return Term(f"{self.coefficient - other.coefficient}x^{self.power}")
        else:
            raise ValueError("Cannot subtract terms with different powers")

    def __truediv__(self, other) -> Term:
        return Term(f"{self.coefficient / other.coefficient}x^{self.power - other.power}")

    def __neg__(self) -> Term:
        return Term(f"{-self.coefficient}x^{self.power}")

    def evaluate(self,n):
        """
        Evaluates value of a term at x=n
        :param n: value of x to evaluate
        :return: the value of term at x=n
        """
        return self.coefficient*(n**self.power)

    @staticmethod
    def arrange(terms: list[Term]) -> list[Term]:
        """
        Arranges a given list of terms in decreasing order of power.
        :param terms: a list of terms
        :return: a list of terms arranged in decreasing order of power
        """
        return sorted(terms, key=lambda term: term.power, reverse=True)


    def __abs__(self) -> Term:
        a = Term("1x^1")
        a.coefficient = abs(self.coefficient)
        a.power = self.power
        return a

    def __str__(self) -> str:
        return self.face

    def __repr__(self) -> str:
        return self.face

class Polynomial:
    """
    This class is for working with mathematical polynomials.
    Note: Polynomials cannot have division of two expressions.

    Class Methods:
        - zero

    Class Properties:
        - face

    Instance Methods:
        - add
        - multiply
        - subtract

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
        logging.debug("expression before cleaning: %s", expression)
        expression = expression.lower()
        allowed_symbols = ["+","-","*","/","^","(",")","[","]","|","0","1","2","3","4","5","6","7","8","9", "x"]
        expression = expression.replace(" ", "")
        if "x" not in expression:
            raise FormatError(msg="THERE HAS TO BE ONE VARIABLE!?!!")
        for i in expression:
            if i not in allowed_symbols:
                raise FormatError(msg="Formating of the given expression is not permitted in the grounds of the polynomials")


        expression = Polynomial._clean(expression)
        terms = Polynomial.exp_eval(expression)
        self.terms = Term.arrange(Polynomial.shorten(terms))

    @classmethod
    def zero(cls):
        obj = cls.__new__(cls)
        obj.terms = []
        return obj

    @property
    def face(self) -> str:
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

    @staticmethod
    def shorten(nums:list[Term]) -> list[Term]:
        """
            Simplifies a given list of terms by adding terms with same power.
            :param nums: a list of terms
            :return: a simplified list of terms with all terms having unique powers
            """
        powers = {}
        for num in nums:
            if num.power not in powers:
                powers[num.power] = num
            else:
                powers[num.power] += num

        result = [powers[key] for key in powers]
        return result

    @staticmethod
    def _clean(expr: str) -> str:
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
    def _implicate_multiplication(expr: str) -> str:
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
                        and (ch == "(" or ch == "x" or ch.isdigit())
                ):
                    result.append("*")
            result.append(ch)
            prev = ch

        return "".join(result)

    @staticmethod
    def _single_layer_split(expr: str, ops: set[str]) -> list[str]:
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
    def _remove_outer_brackets(expr: str) -> str:
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
        expr = Polynomial._implicate_multiplication(expr)

        # remove outer brackets
        sign, expr = _leading_unary(expr)
        expr = Polynomial._remove_outer_brackets(expr)

        # + / -
        parts = Polynomial._single_layer_split(expr, {"+", "-"})
        if len(parts) > 1:
            if sign == -1:
                parts[0] = "-" + parts[0]
            terms = []
            for part in parts:
                terms += Polynomial.exp_eval(part)
            return Polynomial.shorten(terms)

        # *
        factors = Polynomial._single_layer_split(expr, {"*"})
        if len(factors) > 1:
            result = Polynomial.exp_eval(factors[0])
            for f in factors[1:]:
                result = Term.group_multiply(result, Polynomial.exp_eval(f))
                result = Term(f"{sign}x^0").multiply(result)
            return Polynomial.shorten(result)

        # base case
        try:
            return [Term(expr)*sign]
        except FormatError:
            r = Rat(expr)
            return [Term(f"{r}x^0")*sign]

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

        simplified_terms = Polynomial.shorten(new_terms)
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
            result_poly.terms = Term.arrange(Polynomial.shorten(new_terms))
            return result_poly
        elif isinstance(other, Rat) or isinstance(other, int):
            new_terms = Term.group_multiply(self.terms, [Term(f"{other}x^0")])
            result_poly = Polynomial("0")
            result_poly.terms = Term.arrange(Polynomial.shorten(new_terms))
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

    def __str__(self):
        return self.face
    def __repr__(self):
        return self.face

if __name__ == "__main__":
    print("This is a module for rational numbers, terms and polynomials.")
    print("Import this module to use its classes and methods.")