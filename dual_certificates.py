"""Tools for creating interpolation-based dual certificates."""

import numpy as np
from kernels import TrigPoly


def interpolate(support, sign_pattern, kernel):
    assert support.shape == sign_pattern.shape
    assert np.all(np.absolute(np.absolute(sign_pattern) - 1.0) < 1e-10)

    n = support.shape[0]

    # time_deltas[i, j] = t_i - t_j
    time_deltas = np.outer(support, np.ones(n)) - np.outer(np.ones(n), support)
    kernel_values = kernel(time_deltas)

    coeffs = np.linalg.solve(kernel_values, sign_pattern)

    return sum(
        [kernel.shift(-t) * c for c, t in zip(coeffs, support)],
        TrigPoly.zero())


def interpolate_with_derivative(support, sign_pattern, kernel):
    assert support.shape == sign_pattern.shape
    assert np.all(np.absolute(np.absolute(sign_pattern) - 1.0) < 1e-10)

    n = support.shape[0]

    kernel1 = kernel.derivative()
    kernel2 = kernel1.derivative()

    # time_deltas[i, j] = t_i - t_j
    time_deltas = np.outer(support, np.ones(n)) - np.outer(np.ones(n), support)
    kernel_values = kernel(time_deltas)
    kernel1_values = kernel1(time_deltas)
    kernel2_values = kernel2(time_deltas)

    problem_mx = np.bmat([
        [kernel_values, kernel1_values],
        [kernel1_values, kernel2_values]])
    problem_obj = np.hstack([sign_pattern, np.zeros(sign_pattern.shape[0])])

    coeffs = np.linalg.solve(problem_mx, problem_obj)

    return (
        sum([kernel.shift(-t) * c for c, t in zip(coeffs[:n], support)],
            TrigPoly.zero()) +
        sum([kernel1.shift(-t) * c for c, t in zip(coeffs[n:], support)],
            TrigPoly.zero()))
