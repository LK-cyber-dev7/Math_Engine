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


class Counter:
    """
    A class for cycling through a list of values. The values could be a range or a list/tuple.
    Counter object is immutable in the sense that the original range or dataset is a tuple which can not be changed.
    Note: Iterating through a Counter object or cycling through a Counter object using .cycle does NOT change the original pointer
    Note: All number datatypes except floats are considered to not have imperfections during calculations.
    Floats are rounded to 12 decimal places to handle imperfections.
    Note: Start and End are both inclusive.

    Class methods:
        - duplicate

    Class properties:
        - value
        - pointer

    Instance Methods:
        - advance
        - back
        - jump
        - peek
        - reset
        - cycle

    """

    def __init__(self, stop, start=0, step=1):
        if any(isinstance(x, bool) for x in (start, stop, step)):
            raise TypeError("Bool values cannot act as start, stop or step.")
        if isinstance(stop, (list, tuple)):
            if not stop:
                raise ValueError("Empty datasets are not allowed")
            if not isinstance(start, int) or not isinstance(step, int):
                raise TypeError('start and step must be integers')

            self.__data = stop[start::step]
            if not self.__data:
                raise ValueError("Resulting Dataset is empty.")

        else:
            uses_float = any(isinstance(x, float) for x in (start, stop, step))
            if not uses_float:
                Counter._validate_numeric(start=start, stop=stop, step=step)
                self.__data = []
                i = start
                if step > 0:
                    while i <= stop:
                        self.__data.append(i)
                        i += step
                else:
                    while i >= stop:
                        self.__data.append(i)
                        i += step
            else:
                self.__data = []
                eps = 1e-12
                i = start
                Counter._validate_numeric(start=start, stop=stop, step=step)
                if step > 0:
                    while i <= stop + eps:
                        self.__data.append(round(i, 12))
                        i += step
                else:
                    while i >= stop - eps:
                        self.__data.append(round(i, 12))
                        i += step

        self.__data = tuple(self.__data)

        self.__pointer = 0
        self.__iter_point = 0

    @classmethod
    def duplicate(cls, count_obj: Counter) -> Counter:
        """
        Create a copy of the given Counter object.
        :param count_obj: Counter object to be copied
        :return: The copied object.
        """
        obj = cls.__new__(cls)
        obj.__data = count_obj.__data
        obj.__pointer = count_obj.__pointer
        obj.__iter_point = 0
        return obj

    @staticmethod
    def _validate_numeric(start, stop, step):
        if step == 0:
            raise ValueError('step must be non-zero')
        if abs(stop - start) < abs(step):
            raise ValueError("Inappropriate step given")
        if (stop < start and step > 0) or (stop > start and step < 0):
            raise ValueError("Inappropriate step given")

    @property
    def value(self):
        return self.__data[self.__pointer]

    @property
    def pointer(self) -> int:
        return self.__pointer

    def advance(self) -> None:
        """
        Moves the pointer forward.
        """
        if self.__pointer < len(self.__data) - 1:
            self.__pointer += 1
        else:
            self.__pointer = 0

    def back(self) -> None:
        """
        Moves the pointer backward.
        """
        if self.__pointer > 0:
            self.__pointer -= 1
        else:
            self.__pointer = len(self.__data) - 1

    def reset(self) -> None:
        """
        Resets the pointer to 0.
        """
        self.__pointer = 0

    def jump(self, n: int) -> None:
        """
        Jumps the pointer directly to n.
        :param n: Index to where the pointer should jump.
        """
        if not isinstance(n, int):
            raise TypeError('n must be integer')
        self.__pointer = n % len(self.__data)

    def peek(self, n: int):
        """
        Returns value at specific index.
        :param n: Index to peek value at.
        :return: The value at index = n
        """
        if not isinstance(n, int):
            raise TypeError('n must be integer')

        return self.__data[n % len(self.__data)]

    def cycle(self):
        """
        A generator function that cycles through the dataset forever
        :yield: The next value.
        """
        self.__iter_point = 0
        while True:
            yield self.__data[self.__iter_point]
            if self.__iter_point < len(self.__data) - 1:
                self.__iter_point += 1
            else:
                self.__iter_point = 0

    def __getitem__(self, idx: int):
        return self.__data[idx]

    def __eq__(self, other) -> bool:
        return (
                isinstance(other, Counter) and
                self.__data == other.__data and
                self.__pointer == other.__pointer
        )

    def __iter__(self):
        new_object = Counter.duplicate(self)
        new_object.__iter_point = 0
        return new_object

    def __next__(self):
        if self.__iter_point > len(self.__data) - 1:
            raise StopIteration
        val = self.__data[self.__iter_point]
        self.__iter_point += 1
        return val

    def __len__(self):
        return len(self.__data)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"Counter(value={self.value}, pointer={self.__pointer})"