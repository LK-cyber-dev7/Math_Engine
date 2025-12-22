"""
This module can be used for working with rational numbers, terms, and polynomials.
"""
# imports
from __future__ import annotations
import math
from decimal import Decimal


class FormatError(Exception):
    """
    Exception raised when formatting does match to what is expected.
    """
    def __init__(self, msg="String formating is NOT ok"):
        super(FormatError, self).__init__(msg)
        self.msg = msg

def ftr(n: float) -> Rat:
    """
    Converts a floating point number into a Rational number.
    :param n: a float
    :return: The equivalent Rational number in the form of a Rat object.
    """
    # getting number of decimal places
    num = Decimal(str(n))
    dp = num.as_tuple().exponent * -1
    # converting the float to int by removing the point
    p = int(n*(10**int(dp)))
    # getting the denominator by raising 10 to the power equal to dp
    q = int(10**int(dp))
    return Rat(f"[{p}|{q}]").simplify()

def ftr_rep(b: float, a: float =0):
    """
    Converts a repeating floating point number into a Rational number.
    :param b: the repeating part in the form of a float.For example 0.123123... b=0.123
    :param a: if the number starts to repeat after a certain digit 'a' is the float till that digit
    :return: The equivalent Rational number in the form of a Rat object.
    """
    if b <= 0 or b >= 1:
        raise ValueError("b is out of range")
    n1 = Decimal(str(a))
    dp_a = int(n1.as_tuple().exponent * -1)
    # getting the non repeat part
    first = ftr(a)
    # getting number of digits in b
    n2 = Decimal(str(b))
    dp_b = int(n2.as_tuple().exponent * -1)
    # converting b to a rational number by setting numerator to the integer version of b and
    # setting denominator to 9 repeating the same number of times as the number of digits in b
    repeat = Rat(f"[{str(b).replace(".","")[1:]}|{"9"*dp_b+"0"*dp_a}]")
    # returning a+b (non-repeating part plus the repeating part)
    result = first+repeat
    return result.simplify()

def _parse_rat(x: str) -> tuple[int, int]:
    try:
        int(x)
        p = int(x)
        q = 1

    except ValueError:
        if "[" not in x or "]" not in x or "|" not in x:
            raise FormatError
        else:
            if x[0] == "-":
                x = x[1:]
                sep = x.index("|")
                p = int(x[1:sep]) * -1
                q = int(x[sep + 1:-1])
            elif x[0] == "+":
                x = x[1:]
                sep = x.index("|")
                p = int(x[1:sep])
                q = int(x[sep + 1:-1])
            else:
                sep = x.index("|")
                p = int(x[1:sep])
                q = int(x[sep + 1:-1])

    if q < 0:
        p = p
        q = q
    else:
        p = p
        q = q

    return p, q

class Rat:
    """Class for working with rational numbers.

    This class can be used for working with and manipulating rational numbers.
    To create a Rat object pass a string in the form of [p|q]. Notice that the middle separator is a vertical line.
    p represents the numerator and q represents the denominator.
    Objects of Rat can be used as numbers for comparison,assignment and arithmetic operators.

    Class methods:
        simplify
        reciprocal
        add
        multiply
        produce
        eql
    """

    def __init__(self,x: str | Rat | int):

        self.p,self.q = _parse_rat(str(x))

        self.value = self.p/self.q

        self.face = f"[{self.p}|{self.q}]"

    def simplify(self) -> Rat:
        """
        Simplifies rational numbers.
        :return: A simplified Rat object.
        """
        high = math.gcd(self.p, self.q)
        return Rat(f"[{self.p//high}|{self.q//high}]")

    def reciprocal(self) -> Rat:
        """
        :return: Reciprocal of self in form of a Rat object.
        """
        return Rat(f"[{self.q}|{self.p}]")

    def add (self,other: Rat) -> Rat:
        """
        Adds two rational numbers.
        :param other: A Rat object.
        :return: The sum in form of a Rat object.
        """

        if self.q == other.q:
            return Rat(f"[{self.p + other.p}|{self.q}]")
        else:
            new_q = math.lcm(self.q,other.q)
            new_p = int(self.p*(new_q//self.q)+other.p*(new_q//other.q))
            return Rat(f"[{new_p}|{new_q}]")

    def multiply(self,other: Rat) -> Rat:
        """
        Multiplies two rational numbers.
        :param other: A Rat object.
        :return: The product in form of a Rat object.
        """
        if isinstance(other, int):
            other = Rat(other)
        return Rat(f"[{self.p*other.p}|{self.q*other.q}]")

    def produce(self,n:int) -> Rat:
        """
        Returns an equivalent rational number by multiplying n (The integer) to both numerator and denominator.
        :param n: an integer to be multiplied by the rational number.
        :return: The equivalent rational number in form of a Rat object.
        """
        return Rat(f"[{self.p*n}|{self.q*n}]")

    def eql(self,other: Rat) -> tuple[Rat, Rat]:
        """
        Returns a tuple of the given rational number and self with same denominator.
        :param other: The other rational number.
        :return: A tuple of the two rational numbers
        """
        new_q = math.lcm(self.q,other.q)
        a = Rat(f"[{self.p*(new_q//self.q)}|{new_q}]")
        b = Rat(f"[{other.p*(new_q//other.q)}|{new_q}]")

        return a,b

    def __add__(self, other):
        if isinstance(other, Rat):
            return self.add(other).simplify()
        else:
            return self.value + other

    def __radd__(self, other):
        return self.add(other)

    def __mul__(self, other):
        if isinstance(other, Rat):
            return self.multiply(other).simplify()
        else:
            return self.value * other

    def __rmul__(self, other):
        return self.multiply(other)

    def __neg__(self):
        return self.multiply(Rat(-1))

    def __pos__(self):
        return self

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -other + self

    def __truediv__(self, other):
        return self.__mul__(other.reciprocal())

    def __rtruediv__(self, other):
        return self.reciprocal().multiply(other)

    def __abs__(self):
        return Rat(f"[{abs(self.p)}|{abs(self.q)}]")

    def __pow__(self, power, modulo=None):
        if isinstance(power, int):
            return Rat(f"[{int(self.p**power)}|{int(self.q**power)}]")
        else:
            return self.value ** power

    def __eq__(self, other):
        tup = self.eql(other)
        return tup[0].p == tup[1].p

    def __lt__(self, other):
        tup = self.eql(other)
        return tup[0].p < tup[1].p

    def __le__(self, other):
        tup = self.eql(other)
        return tup[0].p <= tup[1].p

    def __gt__(self, other):
        tup = self.eql(other)
        return tup[0].p > tup[1].p

    def __ge__(self, other):
        tup = self.eql(other)
        return tup[0].p >= tup[1].p

    def __ne__(self, other):
        tup = self.eql(other)
        return tup[0].p != tup[1].p

    def __getitem__(self, item):
        if isinstance(item, int):
            return (self.p, self.q)[item]
        else:
            if item.lower() == "p":
                return self.p
            elif item.lower() == "q":
                return self.q
            else:
                raise IndexError

    def __round__(self, n=None):
        return round(self.value, n)

    def __floor__(self):
        return math.floor(self.value)

    def __float__(self):
        return self.value

    def __int__(self):
        return math.floor(self.value)

    def __str__(self):
        return self.face

class Term:
    """Class for working with terms.
    This class can be used for working with and manipulating terms.
    To create an object of Term pass a string which includes a variable x.
    Coefficients can be int or Rat while powers of x can only be int.To represent exponentiation use '^' instead of '**'

    Term methods:
    is_equal
    simplify
    multiply
    evaluate
    """

    def __init__(self, x: str):
        for i in x:
            if i in ["(",")","{","}"]:
                raise FormatError

        if x.count("x") != 1:
            raise FormatError

        sep = x.index("x")
        if x[0] == "x":
            self.coefficient = Rat(1)
        else:
            self.coefficient = Rat(x[:sep])
        self.varipart = x[sep:]

        if "^" in self.varipart:
            sep = self.varipart.index("^")
            self.power = int(self.varipart[sep+1:])

        elif x[-1] != "x":
            raise FormatError

        else:
            self.power = 1

        self.face = f"{self.coefficient}x^{self.power}"

    def is_equal(self, other: Term) -> bool:
        """
        Returns True if both the terms are of same power
        :param other: The second term to be compared
        :return: True if both terms are of same power otherwise False
        """
        return self.power == other.power

    def __add__(self, other):
        if self.power == other.power:
            return Term(f"{self.coefficient + other.coefficient}x^{self.power}")
        else:
            return None

    def __mul__(self, other):
        if isinstance(other,Term):
            return Term(f"{self.coefficient * other.coefficient}x^{self.power+other.power}")
        elif isinstance(other,Rat):
            return Term(f"{self.coefficient * other}x^{self.power}")
        else:
            return None

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


    def __sub__(self, other):
        if self.power == other.power:
            return Term(f"{self.coefficient - other.coefficient}x^{self.power}")
        else:
            return None

    def __truediv__(self, other):
        return Term(f"{self.coefficient / other.coefficient}x^{self.power - other.power}")

    def evaluate(self,n):
        """
        Evaluates value of a term at x=n
        :param n: value of x to evaluate
        :return: the value of term at x=n
        """
        return self.coefficient*(n**self.power)

    def __str__(self):
        return self.face

def _shorten(nums:list[Term]) -> list[Term]:
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

def exp_eval(item: str) -> list[Term]:
    """
    Simplifies a given expression by multiplying all the brackets.
    :param item: an expression in form of a string
    :return: a list of terms after all the multiplication of brackets and addition of terms.
    """
    # removing spaces to avoid later errors
    item = item.replace(" ","") # expression
    things = [] # list of the terms in the form of strings
    mult = [] # list of the terms in the form Term objects and lists of Term objects
    brac = [0, 0] # a list to keep track of opened and closed brackets
    sep = 0 # variable to store the index which separates the terms

    # separating the string based on brackets.
    for i in range(len(item)):
        # checking if the bracket is opening
        if item[i] == "(":
            # if all previously opened brackets are closed then do this
            if brac[0] == brac[1]:
                things.append(item[sep:i])
                sep = i
                brac[0] += 1
            # else if there is a bracket inside a bracket then just increase the number of opened brackets
            # this will be picked up later to raise a "Too many layered brackets" error
            else:
                brac[0] += 1
        # same if the brackets are being closed
        elif item[i] == ")":
            if brac[0] - 1 == brac[1]:
                things.append(item[sep:i+1])
                sep = i+1
                brac[1] += 1
            else:
                brac[1] += 1
    # if in the last there is a term outside the bracket, the following will add that
    if sep != len(item):
        things.append(item[sep:])

    # now we convert all the terms into Term objects
    for tng in things:
        if len(tng) == 0:
            continue
        tng = tng.replace(" ","")
        # removing ending brackets like if tng="(3x-4x^2)" then it will equal "3x-4x^2"
        if tng[0] == "(" and tng[-1] == ")":
            tng = tng[1:-1]

        dead = False # variable to check if tng is an expression or a single term
        # we try to convert to a Rat object just to check
        try:
            trl = Rat(tng)
        # if it is not a rat object then we try Term object
        except (FormatError, ValueError):
            # we add the term object to the mult list
            try:
                mult.append(Term(tng))
            # if tng is not a term or a rat we set dead=True which later will be checked
            except FormatError:
                dead = True
        # if it was converted into Rat without error then we convert it into a term object with x raised to 0
        else:
            mult.append(Term(f"{str(trl)}x^0"))


        # if it is neither term nor a rat then we check for brackets
        if dead:
            a = [] # this is the list that is later added to mult
            # if there are brackets in tng that indicates that there were more than 1 layer of brackets
            # as we don't want brackets inside brackets we raise an error
            if "(" in tng or ")" in tng:
                raise FormatError("Too many layered brackets. Program can't handle.")
            # once checked for layered brackets we separate tng based on + and -
            # then we convert each one into a term and add all the terms in a list
            # that list is added to mult
            else:
                sep = 0
                # separation based on + and - signs
                for i in range(len(tng)):
                    if tng[i] in ["+","-"]:
                        # checking if it is a term
                        try:
                            a.append(Term(tng[sep:i]))
                        # if not we try with a rat nd then convert to a term with x raised to 0
                        except FormatError:
                            try:
                                trial = Rat(tng[sep:i])
                            # if term and rat both fail we raise an error
                            except FormatError:
                                raise FormatError("Error occurred in initializing polynomial")
                            else:
                                a.append(Term(f"{str(trial)}x^0"))
                        sep = i
                # when we separate based on + and - the code skips the last term
                # here we try to convert it into a term or a rat.
                # if neither works, we raise error
                try:
                    a.append(Term(tng[sep:]))
                except FormatError:
                    try:
                        trial = Rat(tng[sep:])
                    except FormatError:
                        raise FormatError("Error occurred in initializing polynomial")
                    else:
                        a.append(Term(f"{str(trial)}x^0"))
                mult.append(a)

    # now we have a list of things that are to be multiplied
    # mult only contains two types of things either a term or a list of terms
    # this loop repeats one once for every item in the list except the first one
    # every time the loop is run the value of mult[0] is set to  product of mult[0] and mult[1]
    num = len(mult)-1
    for _ in range(num):
        # if mult[0] is a term then multiply and set mult[0] to the product
        if isinstance(mult[0],Term):
            mult[0] = mult[0].multiply(mult[1])
            # removing the number once it is multiplied
            mult.remove(mult[1])
        # if it is a list then loop through the list and multiply one by one.
        elif isinstance(mult[0],list):
            ans = []
            for j in mult[0]:
                k = j.multiply(mult[1])
                if isinstance(k,Term):
                    ans.append(j.multiply(mult[1]))
                else:
                    ans += k


            mult[0] = ans
            # removing the list once it is multiplied
            mult.remove(mult[1])
    # mult is simplified using _shorten and then returned.
    return _shorten(mult[0])

class Polynomial:
    """
    This class is for working with mathematical polynomials.
    Note: Polynomials cannot have division of two expressions.

    class methods:
    -
    """
    def __init__(self,expression:str):
        allowed_symbols = ["+","-","*","/","^","(",")","[","]","|","0","1","2","3","4","5","6","7","8","9","x"]
        expression = expression.replace(" ", "")
        expression = expression.replace("*", "")
        if "x" not in expression:
            raise FormatError(msg="THERE HAS TO BE ONE VARIABLE!?!!")
        for i in expression:
            if i not in allowed_symbols:
                raise FormatError(msg="Formating of the given expression is not permitted in the grounds of the polynomials")


        self.exp = expression
        brac = [0,0]
        sep = 0
        parts = []
        for t in range(len(self.exp)):
            item = self.exp[t]
            if item == "(":
                brac[0] += 1
            elif item == ")":
                brac[1] += 1
            elif item in ["+","-"] and brac[0] == brac[1]:
                parts.append(self.exp[sep:t])
                sep = t
        parts.append(self.exp[sep:])

        terms = []
        upt = []

        for i in range(len(parts)):
            part = parts[i]
            if "(" not in part and ")" not in part:
                try:
                    terms.append(Term(part))
                except FormatError:
                    try:
                        terms.append(Rat(part))
                    except FormatError:
                        pass

            elif "(" in part and ")" in part:
                upt.append(part)
            else:
                try:
                    terms.append(Rat(part))
                except FormatError:
                    pass

        if len(parts) != len(upt) + len(terms):
            raise FormatError(msg="Terms Could not be processed")

        if len(upt) != 0:
            for item in upt:
                pass

