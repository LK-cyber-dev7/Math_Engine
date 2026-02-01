import math

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

def unique(values, tolerance):
    result = []
    for v in values:
        if not any(abs(v - u) < tolerance for u in result):
            result.append(v)
    return result