from positive_spoof_factorizations import (
    partialPositiveSpoofLehmerFactorization,
    compute_product_of_list,
    evaluate,
    compute_totient,
    yield_all_positive_spoof_Lehmer_factorizations_given_r_k_parity,
)
from fractions import Fraction
from collections import namedtuple
import numpy as np
import pytest
from itertools import islice


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
        TestCase(numbers=[]),
        TestCase(numbers=[i for i in range(1, 5)]),
        TestCase(numbers=[i for i in range(-17, 5)]),
        TestCase(numbers=("ABA",)),
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
        TestCase(numbers=[]),
        TestCase(numbers=[i + 1 for i in range(1, 5)]),
        TestCase(numbers=[i + 1 for i in range(-17, 5)]),
        TestCase(numbers=("ABA",)),
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
        partialPositiveSpoofLehmerFactorization(2, 2, [3]),
        partialPositiveSpoofLehmerFactorization(2, 2, [3, 3]),
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
    input_list = [
        TestCase(
            spoof=partialPositiveSpoofLehmerFactorization(2, 2, [3]),
            additional_factors=3,
        )
    ]
    output_list = [partialPositiveSpoofLehmerFactorization(2, 2, [3, 3])]
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


def test_yield_all_spoof_Lehmer_factorizations_given_r_k_parity():
    """
    Tests the yield_all_spoof_Lehmer_factorizations_given_r_k_parity method.
    """

    function_to_test = yield_all_positive_spoof_Lehmer_factorizations_given_r_k_parity

    # Create a namedtuple to better organize test case parameters
    TestCase = namedtuple(
        "TestCase",
        [
            "r",
            "k",
            "base_spoof",
            "is_even"
        ],
    )

    input_list = [
        TestCase(r=2, k=2, base_spoof=None, is_even=False),
    ]
    output_list = [
        [partialPositiveSpoofLehmerFactorization(2, 0, 2, [3, 3])],
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
