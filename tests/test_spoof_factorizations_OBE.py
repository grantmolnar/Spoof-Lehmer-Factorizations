from spoof_factorizations_OBE import (
    partialSpoofLehmerFactorization,
    count_positives_negatives,
    sort_by_magnitude_then_positivity,
    integer_magnitude_iterator,
    compute_product_of_list,
    evaluate,
    compute_totient,
    yield_all_spoof_Lehmer_factorizations_given_rplus_rminus_k,
)
from fractions import Fraction
from collections import namedtuple
import numpy as np
import pytest
from itertools import islice


def test_count_positives_negatives():
    """
    Tests the count_positives_negatives method.
    """

    function_to_test = count_positives_negatives

    # Create a namedtuple to better organize test case parameters
    TestCase = namedtuple(
        "TestCase",
        [
            "numbers",
        ],
    )

    input_list = [
        TestCase([]),
        TestCase([i for i in range(1, 5)]),
        TestCase([i for i in range(-17, 5)]),
        TestCase(("ABA",)),
    ]
    output_list = [
        (0, 0),
        (4, 0),
        (4, 17),
        AssertionError,
    ]
    assert len(input_list) == len(output_list)
    for test_input, ostensible_output in zip(input_list, output_list):
        if isinstance(ostensible_output, type) and issubclass(
            ostensible_output, Exception
        ):
            # Test case is expected to raise an exception
            with pytest.raises(ostensible_output):
                function_to_test(*test_input)
        else:
            test_output = function_to_test(*test_input)
            # print(f"Test Input: {test_input}\nTest Output: {test_output}\nOstensible Output: {ostensible_output}")
            assert test_output == ostensible_output
            # assert abs(test_output - ostensible_output) < epsilon


def test_sort_by_magnitude_then_positivity():
    """
    Tests the sort_by_magnitude_then_positivity method.
    """

    function_to_test = sort_by_magnitude_then_positivity

    # Create a namedtuple to better organize test case parameters
    TestCase = namedtuple(
        "TestCase",
        [
            "numbers",
        ],
    )

    input_list = [
        TestCase([]),
        TestCase([i for i in range(1, 5)]),
        TestCase([i for i in range(4, -1, -1)]),
        TestCase([i for i in range(-17, 5)]),
        TestCase(("ABA",)),
    ]
    output_list = [
        [],
        [1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [
            0,
            1,
            -1,
            2,
            -2,
            3,
            -3,
            4,
            -4,
            -5,
            -6,
            -7,
            -8,
            -9,
            -10,
            -11,
            -12,
            -13,
            -14,
            -15,
            -16,
            -17,
        ],
        TypeError,
    ]
    assert len(input_list) == len(output_list)
    for test_input, ostensible_output in zip(input_list, output_list):
        if isinstance(ostensible_output, type) and issubclass(
            ostensible_output, Exception
        ):
            # Test case is expected to raise an exception
            with pytest.raises(ostensible_output):
                function_to_test(*test_input)
        else:
            test_output = function_to_test(*test_input)
            # print(f"Test Input: {test_input}\nTest Output: {test_output}\nOstensible Output: {ostensible_output}")
            assert test_output == ostensible_output
            # assert abs(test_output - ostensible_output) < epsilon


def test_integer_magnitude_iterator():
    """
    Tests the integer_magnitude_iterator method.
    """

    function_to_test = integer_magnitude_iterator

    # Create a namedtuple to better organize test case parameters
    TestCase = namedtuple(
        "TestCase",
        ["start", "step"],
    )

    N = 6

    input_list = [
        TestCase(1, 3),
        TestCase(0, 4),
        TestCase(-1, 1),
        TestCase(2, 0),
        TestCase("ABA", None),
    ]
    output_list = [
        [1, -1, 4, -4, 7, -7],
        [0, 4, -4, 8, -8, 12],
        [-1, 2, -2, 3, -3, 4],
        AssertionError,
        TypeError,
    ]
    assert len(input_list) == len(output_list)
    for test_input, ostensible_output in zip(input_list, output_list):
        if isinstance(ostensible_output, type) and issubclass(
            ostensible_output, Exception
        ):
            # Test case is expected to raise an exception
            with pytest.raises(ostensible_output):
                test_output = list(islice(function_to_test(*test_input), N))
        else:
            test_output = list(islice(function_to_test(*test_input), N))
            # print(f"Test Input: {test_input}\nTest Output: {test_output}\nOstensible Output: {ostensible_output}")
            assert test_output == ostensible_output
            # assert abs(test_output - ostensible_output) < epsilon


def test_compute_product_of_list():
    """
    Tests the compute_product_of_list method.
    """

    function_to_test = compute_product_of_list

    # Create a namedtuple to better organize test case parameters
    TestCase = namedtuple(
        "TestCase",
        [
            "numbers",
        ],
    )

    input_list = [
        TestCase([]),
        TestCase([i for i in range(1, 5)]),
        TestCase([i for i in range(-17, 5)]),
        TestCase(("ABA",)),
    ]
    output_list = [
        1,
        24,
        0,
        AssertionError,
    ]
    assert len(input_list) == len(output_list)
    for test_input, ostensible_output in zip(input_list, output_list):
        if isinstance(ostensible_output, type) and issubclass(
            ostensible_output, Exception
        ):
            # Test case is expected to raise an exception
            with pytest.raises(ostensible_output):
                function_to_test(*test_input)
        else:
            test_output = function_to_test(*test_input)
            # print(f"Test Input: {test_input}\nTest Output: {test_output}\nOstensible Output: {ostensible_output}")
            assert test_output == ostensible_output
            # assert abs(test_output - ostensible_output) < epsilon


def test_evaluate():
    """
    Tests the evaluate method.
    """

    function_to_test = evaluate

    # Create a namedtuple to better organize test case parameters
    TestCase = namedtuple(
        "TestCase",
        [
            "numbers",
        ],
    )

    input_list = [
        TestCase([]),
        TestCase([i for i in range(1, 5)]),
        TestCase([i for i in range(-17, 5)]),
        TestCase(("ABA",)),
    ]
    output_list = [
        1,
        24,
        0,
        AssertionError,
    ]
    assert len(input_list) == len(output_list)
    for test_input, ostensible_output in zip(input_list, output_list):
        if isinstance(ostensible_output, type) and issubclass(
            ostensible_output, Exception
        ):
            # Test case is expected to raise an exception
            with pytest.raises(ostensible_output):
                function_to_test(*test_input)
        else:
            test_output = function_to_test(*test_input)
            # print(f"Test Input: {test_input}\nTest Output: {test_output}\nOstensible Output: {ostensible_output}")
            assert test_output == ostensible_output
            # assert abs(test_output - ostensible_output) < epsilon


def test_compute_totient():
    """
    Tests the compute_totient method.
    """

    function_to_test = compute_totient

    # Create a namedtuple to better organize test case parameters
    TestCase = namedtuple(
        "TestCase",
        [
            "numbers",
        ],
    )

    input_list = [
        TestCase([]),
        TestCase([i + 1 for i in range(1, 5)]),
        TestCase([i + 1 for i in range(-17, 5)]),
        TestCase(("ABA",)),
    ]
    output_list = [
        1,
        24,
        0,
        TypeError,
    ]
    assert len(input_list) == len(output_list)
    for test_input, ostensible_output in zip(input_list, output_list):
        if isinstance(ostensible_output, type) and issubclass(
            ostensible_output, Exception
        ):
            # Test case is expected to raise an exception
            with pytest.raises(ostensible_output):
                function_to_test(*test_input)
        else:
            test_output = function_to_test(*test_input)
            # print(f"Test Input: {test_input}\nTest Output: {test_output}\nOstensible Output: {ostensible_output}")
            assert test_output == ostensible_output
            # assert abs(test_output - ostensible_output) < epsilon


def test_k_bounds():
    """
    Tests the k_bounds method.
    """

    input_list = [
        partialSpoofLehmerFactorization(2, 0, 2, [3]),
        partialSpoofLehmerFactorization(2, 0, 2, [3, 3]),
    ]
    output_list = [(Fraction(3, 2), Fraction(2, 1)), (Fraction(2, 1), Fraction(2, 1))]
    assert len(input_list) == len(output_list)
    for test_input, ostensible_output in zip(input_list, output_list):
        if isinstance(ostensible_output, type) and issubclass(
            ostensible_output, Exception
        ):
            # Test case is expected to raise an exception
            with pytest.raises(ostensible_output):
                test_input.k_bounds()
        else:
            test_output = test_input.k_bounds()
            # print(
            #     f"Test Input: {test_input}\nTest Output: {test_output}\nOstensible Output: {ostensible_output}"
            # )
            assert test_output == ostensible_output
            # assert abs(test_output - ostensible_output) < epsilon


def test_with_additional_factor():
    """
    Tests the with_additional_factor method.
    """
    TestCase = namedtuple(
        "TestCase",
        ["spoof", "additional_factors"],
    )
    input_list = [TestCase(partialSpoofLehmerFactorization(2, 0, 2, [3]), 3)]
    output_list = [partialSpoofLehmerFactorization(2, 0, 2, [3, 3])]
    assert len(input_list) == len(output_list)
    for test_input, ostensible_output in zip(input_list, output_list):
        if isinstance(ostensible_output, type) and issubclass(
            ostensible_output, Exception
        ):
            # Test case is expected to raise an exception
            with pytest.raises(ostensible_output):
                test_input[0].with_additional_factor(test_input[1])
        else:
            test_output = test_input[0].with_additional_factor(test_input[1])
            print(
                f"Test Input: {test_input}\nTest Output: {str(test_output)}\nOstensible Output: {str(ostensible_output)}"
            )
            assert test_output == ostensible_output
            # assert abs(test_output - ostensible_output) < epsilon


def test_yield_all_spoof_Lehmer_factorizations_given_rplus_rminus_k():
    """
    Tests the yield_all_spoof_Lehmer_factorizations_given_rplus_rminus_k method.
    """

    function_to_test = yield_all_spoof_Lehmer_factorizations_given_rplus_rminus_k

    # Create a namedtuple to better organize test case parameters
    TestCase = namedtuple(
        "TestCase",
        [
            "rplus",
            "rminus",
            "k",
            "base_spoof",
        ],
    )

    input_list = [
        TestCase(2, 0, 2, None),
    ]
    output_list = [
        [partialSpoofLehmerFactorization(2, 0, 2, [3, 3])],
    ]
    assert len(input_list) == len(output_list)
    for test_input, ostensible_output in zip(input_list, output_list):
        if isinstance(ostensible_output, type) and issubclass(
            ostensible_output, Exception
        ):
            # Test case is expected to raise an exception
            with pytest.raises(ostensible_output):
                [x for x in function_to_test(*test_input)]
        else:
            test_output = [x for x in function_to_test(*test_input)]
            print(
                f"Test Input: {test_input}\nTest Output: {[str(output) for output in test_output]}\nOstensible Output: {[str(output) for output in ostensible_output]}"
            )
            assert test_output == ostensible_output
            # assert abs(test_output - ostensible_output) < epsilon


# test_yield_all_spoof_Lehmer_factorizations_given_rplus_rminus_k()
