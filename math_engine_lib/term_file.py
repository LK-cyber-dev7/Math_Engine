from __future__ import annotations
from .rat_file import Rat
from .utilities_file import FormatError

def _leading_unary(expr: str) -> tuple[int, str]:
    sign = 1
    i = 0

    while i < len(expr) and expr[i] in "+-":
        if expr[i] == "-":
            sign *= -1
        i += 1

    return sign, expr[i:]

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
