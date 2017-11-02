"""Tools for creating interpolation-based dual certificates."""

import numpy as np
from trig_poly import TrigPoly, MultiTrigPoly


#
# Interpolation functions
#


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


def interpolate_multidim(support, sign_pattern, kernel):
    assert support.shape[0] == sign_pattern.shape[0]
    assert np.all(
        np.absolute(
            np.sum(np.absolute(sign_pattern) ** 2, axis=1) - 1.0) < 1e-10)

    n = support.shape[0]
    m = sign_pattern.shape[1]

    time_deltas = np.outer(support, np.ones(n)) - np.outer(np.ones(n), support)
    kernel_values = kernel(time_deltas)

    coeffss = []
    for k in range(m):
        single_sign_pattern = sign_pattern[:, k]
        coeffss.append(
            np.linalg.solve(kernel_values, single_sign_pattern))

    return MultiTrigPoly([
        sum(
            [kernel.shift(-t) * c for c, t in zip(coeffs, support)],
            TrigPoly.zero())
        for coeffs in coeffss])


#
# Validation functions
#


_EPSILON = 1e-10


def validate(support, sign_pattern, interpolator, grid_pts=1e5):
    values_achieved = True
    for i in range(support.shape[0]):
        if len(sign_pattern.shape) == 1:
            sign_pattern_slice = sign_pattern[i]
        else:
            sign_pattern_slice = sign_pattern[i, :]
        values_achieved = (
            values_achieved and
            np.all(np.absolute(
                interpolator(support[i]) - sign_pattern_slice) < _EPSILON))

    grid = np.linspace(0.0, 1.0, grid_pts)
    grid_values = interpolator(grid)
    if len(grid_values.shape) == 1:
        grid_magnitudes = np.absolute(grid_values)
    else:
        grid_magnitudes = np.linalg.norm(grid_values, axis=0)

    grid_magnitudes = np.ma.array(grid_magnitudes)
    for t in support:
        left_ix = np.searchsorted(grid, t)
        grid_magnitudes[left_ix] = np.ma.masked
        grid_magnitudes[left_ix + 1] = np.ma.masked

    bound_achieved = np.all(grid_magnitudes < 1.0)

    status = values_achieved and bound_achieved

    return {
        'status': status,
        'values_achieved': values_achieved,
        'bound_achieved': bound_achieved}
