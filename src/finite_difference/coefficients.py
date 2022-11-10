import numpy as np


def one_dimensional_coefficients(offsets: list[int | float], order: int) -> dict[int | float, float]:
    matrix = np.vander(offsets, increasing=True).transpose()
    right_hand_side = np.zeros_like(offsets)
    right_hand_side[order] = np.math.factorial(order)
    solution = np.linalg.solve(matrix, right_hand_side)
    coefficients = {offset: solution[index] for index, offset in enumerate(offsets)}
    return coefficients


def one_dimensional_error(coefficients: dict[int | float, float], order: int, max_order: int):
    offsets = np.array(list(coefficients.keys()))
    weights = np.array(list(coefficients.values()))
    power_series = np.vander(offsets, N=max_order, increasing=True)[:, order + 1:]
    term_coefficients = np.array([np.math.factorial(x) for x in np.arange(order + 1, max_order)])
    taylor_polynomials = power_series / term_coefficients
    weighted_terms = taylor_polynomials * weights[:, np.newaxis]
    error = np.sum(weighted_terms, axis=0)
    return error
