from spoof_factorizations import *

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
