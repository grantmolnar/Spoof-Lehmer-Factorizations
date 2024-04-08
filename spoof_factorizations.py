from typing import List, Optional

from functools import reduce
import operator

def product_of_list(numbers: List[int]) -> int:
    """
    Calculates the product of all integers in a given list.
    
    Parameters:
        numbers (List[int]): A list of integers whose product is to be found.
        
    Returns:
        int: The product of all integers in the list. Returns 1 for an empty list.
    
    Example:
        >>> product_of_list([1, 2, 3, 4])
        24
    """
    output = reduce(operator.mul, numbers, 1)
    assert isinstance(output, int)
    return output


def sort_by_magnitude_then_positivity(numbers: List[int]) -> List[int]:
    """
    Sorts a list of integers by their magnitude from least to greatest. In case of a tie,
    positive numbers will come before negative ones.

    Parameters:
        numbers (List[int]): A list of integers to be sorted.

    Returns:
        List[int]: The sorted list of integers, with positive numbers coming before negative
                   numbers in case of a magnitude tie.

    Example:
        >>> sort_by_magnitude_then_positivity([-2, -1, 1, 2, 3, -3])
        [1, -1, 2, -2, 3, -3]
    """
    # Sort primarily by the absolute value, and in case of ties (same magnitude),
    # positive numbers should come before negative ones. This is achieved by
    # checking if the number is negative to influence the secondary sort criteria.
    return sorted(numbers, key=lambda x: (abs(x), 1 if x < 0 else 0))


class partialSpoofLehmerFactorization:
    """
    This class represents a partial spoof Lehmer factorization. It encapsulates four key elements:
    rplus, rminus, an optional k, and an optional list of factors.

    Attributes:
        rplus (int): Represents the positive component of the factorization.
        rminus (int): Represents the negative component of the factorization.
        k (Optional[int]): An optional parameter that can adjust the factorization in some manner.
        factors (Optional[List[int]]): An optional list of integers that are considered factors in this factorization.
    """

    def __init__(
        self,
        rplus: int,
        rminus: int,
        k: Optional[int] = None,
        factors: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes a new instance of the partialSpoofLehmerFactorization class.

        Parameters:
            rplus (int): The positive component of the factorization.
            rminus (int): The negative component of the factorization.
            k (Optional[int]): An optional adjustment parameter, which we'd like to satisfy k * phi(F) = e(F) - 1
            factors (Optional[List[int]]): An optional list of factor integers. Defaults to None.
        """
        self.rplus = rplus
        self.rminus = rminus
        self.k = k
        self.factors = factors if factors is not None else []
        self.evaluation = product_of_list(self.factors)

    def __str__(self) -> str:
        """
        Provides a human-readable string representation of the partialSpoofLehmerFactorization instance.

        Returns:
            str: A string representation of the instance.
        """
        return f"partialSpoofLehmerFactorization(rplus={self.rplus}, rminus={self.rminus}, k={self.k}, factors={self.factors})"
