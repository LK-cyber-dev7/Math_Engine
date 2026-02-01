from __future__ import annotations
from .rat_file import Rat
from .utilities_file import FormatError, unique, find_factors
from .term_file import Term

def _leading_unary(expr: str) -> tuple[int, str]:
    sign = 1
    i = 0

    while i < len(expr) and expr[i] in "+-":
        if expr[i] == "-":
            sign *= -1
        i += 1

    return sign, expr[i:]

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
