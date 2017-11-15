"""Tools for creating interpolation-based dual certificates."""

from trig_poly import MultiTrigPoly

import numpy as np


def _interpolator_norm_quadratic_form(kernel, support):
    """Quadratic form calculating L2 norm of interpolator from coefficients."""
    n = support.shape[0]

    kernel_1 = kernel.derivative()

    kernel_inners = kernel.inners_of_shifts(support)
    kernel_1_inners = kernel_1.inners_of_shifts(support)
    cross_inners = kernel.inners_of_shifts_and_derivative_shifts(support)

    # Build objective quadratic form corresponding to interpolator L2 norm:
    S = np.zeros((4*n, 4*n)).astype(np.complex128)
    S[:n, :n] = kernel_inners
    S[n:2*n, n:2*n] = kernel_1_inners
    S[n:2*n, :n] = cross_inners.T
    S[:n, n:2*n] = cross_inners
    S[2*n:3*n, 2*n:3*n] = kernel_inners
    S[3*n:, 3*n:] = kernel_1_inners
    S[3*n:, 2*n:3*n] = cross_inners.T
    S[2*n:3*n, 3*n:] = cross_inners
    # TODO: Make sure it's ok to cast to real here
    S = (S + S.T).real * 0.5
    return S


def _optimize_quadratic_form(S, A, y):
    """Maximize x^T S_inv x s.t. Ax = y."""
    m = A.shape[0]
    n = S.shape[0]
    return np.linalg.solve(
        np.bmat([[S, A.T], [A, np.zeros((m, m))]]),
        np.hstack([np.zeros(n), y]))[:n]


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

    return kernel.sum_shifts(-support, coeffs)


def interpolate_with_derivative(support, sign_pattern, kernel):
    assert support.shape == sign_pattern.shape
    assert np.all(np.absolute(np.absolute(sign_pattern) - 1.0) < 1e-10)

    n = support.shape[0]

    # time_deltas[i, j] = t_i - t_j
    time_deltas = np.outer(support, np.ones(n)) - np.outer(np.ones(n), support)

    # NOTE: This is assuming that the kernel is real-valued.

    kernel_1 = kernel.derivative()
    kernel_2 = kernel_1.derivative()

    kernel_values = kernel(time_deltas)
    kernel_1_values = kernel_1(time_deltas)
    kernel_2_values = kernel_2(time_deltas)

    sign_pattern_real = np.real(sign_pattern)
    sign_pattern_imag = np.imag(sign_pattern)

    zeros = np.zeros((n, n))

    # Build linear constraint objects
    A = np.bmat([
        [kernel_values, kernel_1_values, zeros, zeros],
        [zeros, zeros, kernel_values, kernel_1_values],
        [sign_pattern_real.reshape((n, 1)) * kernel_1_values,
         sign_pattern_real.reshape((n, 1)) * kernel_2_values,
         sign_pattern_imag.reshape((n, 1)) * kernel_1_values,
         sign_pattern_imag.reshape((n, 1)) * kernel_2_values]]).astype(
             np.float64)
    y = np.hstack(
        [sign_pattern_real,
         sign_pattern_imag,
         np.zeros(sign_pattern.shape[0])])

    # Build objective quadratic form corresponding to interpolator L2 norm:
    S = _interpolator_norm_quadratic_form(kernel, support)

    coeffs = _optimize_quadratic_form(S, A, y)

    return (
        kernel.sum_shifts(-support, coeffs[:n]) +
        kernel_1.sum_shifts(-support, coeffs[n:2*n]) +
        kernel.sum_shifts(-support, coeffs[2*n:3*n] * 1j) +
        kernel_1.sum_shifts(-support, coeffs[3*n:] * 1j))


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
        kernel.sum_shifts(-support, coeffs)
        for coeffs in coeffss])


def interpolate_multidim_with_derivative(
        support, sign_pattern, kernel):
    assert support.shape[0] == sign_pattern.shape[0]
    assert np.all(
        np.absolute(
            np.sum(np.absolute(sign_pattern) ** 2, axis=1) - 1.0) < 1e-10)

    n = support.shape[0]
    m = sign_pattern.shape[1]

    time_deltas = np.outer(support, np.ones(n)) - np.outer(np.ones(n), support)

    kernel_1 = kernel.derivative()
    kernel_2 = kernel_1.derivative()

    kernel_values = kernel(time_deltas)
    kernel_1_values = kernel_1(time_deltas)
    kernel_2_values = kernel_2(time_deltas)

    sign_pattern_real = np.real(sign_pattern)
    sign_pattern_imag = np.imag(sign_pattern)

    #
    # Build linear constraint data
    #

    zeros = np.zeros((n, n))
    problem_mx_rows = []
    problem_obj_cols = []
    for k in range(m):
        # Row of real part constraint
        row1 = []
        for _ in range(4 * k):
            row1.append(zeros)
        row1.append(kernel_values)
        row1.append(kernel_1_values)
        row1.append(zeros)
        row1.append(zeros)
        for _ in range(4 * (m - 1 - k)):
            row1.append(zeros)
        problem_mx_rows.append(row1)

        # Row of imaginary part constraint
        row2 = []
        for _ in range(4 * k):
            row2.append(zeros)
        row2.append(zeros)
        row2.append(zeros)
        row2.append(kernel_values)
        row2.append(kernel_1_values)
        for _ in range(4 * (m - 1 - k)):
            row2.append(zeros)
        problem_mx_rows.append(row2)

    gradient_row = []
    for k in range(m):
        # Row of gradient constraint
        single_sign_pattern_real = sign_pattern_real[:, k]
        single_sign_pattern_imag = sign_pattern_imag[:, k]
        gradient_row.append(
            single_sign_pattern_real.reshape((n, 1)) * kernel_1_values)
        gradient_row.append(
            single_sign_pattern_real.reshape((n, 1)) * kernel_2_values)
        gradient_row.append(
            single_sign_pattern_imag.reshape((n, 1)) * kernel_1_values)
        gradient_row.append(
            single_sign_pattern_imag.reshape((n, 1)) * kernel_2_values)

        # Objective
        problem_obj_cols.append(single_sign_pattern_real)
        problem_obj_cols.append(single_sign_pattern_imag)

    problem_mx_rows.append(gradient_row)
    problem_mx = np.bmat(problem_mx_rows)

    problem_obj_cols.append(np.zeros(n))
    problem_obj = np.hstack(problem_obj_cols)

    #
    # Build objective quadratic form
    #

    # S is block-diagonal, with k blocks of size 4n x 4n each of which is the
    # same as the objective quadratic form from the one-sample case.
    S_diag_block = _interpolator_norm_quadratic_form(kernel, support)
    S = np.kron(np.identity(m), S_diag_block)

    coeffs = _optimize_quadratic_form(S, problem_mx, problem_obj)

    return MultiTrigPoly([
        (kernel.sum_shifts(-support, coeffs[4*k*n:(4*k+1)*n]) +
         kernel_1.sum_shifts(-support, coeffs[(4*k+1)*n:(4*k+2)*n]) +
         kernel.sum_shifts(-support, coeffs[(4*k+2)*n:(4*k+3)*n] * 1j) +
         kernel_1.sum_shifts(-support, coeffs[(4*k+3)*n:(4*k+4)*n] * 1j))
        for k in range(m)])


#
# Validation functions
#


_EPSILON = 1e-8


def validate(support, sign_pattern, interpolator, grid_pts=1e3):
    max_deviation = float('-inf')
    for i in range(support.shape[0]):
        if len(sign_pattern.shape) == 1:
            sign_pattern_slice = sign_pattern[i]
        else:
            sign_pattern_slice = sign_pattern[i, :]
        max_deviation = max(
            max_deviation,
            np.max(
                np.absolute(
                    interpolator(support[i]).T - sign_pattern_slice)))
    values_achieved = max_deviation <= _EPSILON

    grid = np.linspace(0.0, 1.0, grid_pts)
    grid_values = interpolator(grid)
    if len(grid_values.shape) == 1:
        grid_magnitudes = np.absolute(grid_values)
    else:
        grid_magnitudes = np.linalg.norm(grid_values, axis=0)

    grid_magnitudes = np.ma.array(grid_magnitudes)
    for t in support:
        left_ix = np.searchsorted(grid, t)
        grid_magnitudes[left_ix % grid_magnitudes.shape[0]] = np.ma.masked
        grid_magnitudes[(left_ix + 1) % grid_magnitudes.shape[0]] = (
            np.ma.masked)

    bound_achieved = np.all(grid_magnitudes < 1.0)

    status = values_achieved and bound_achieved

    return {
        'status': status,
        'values_achieved': values_achieved,
        'max_deviation': max_deviation,
        'bound_achieved': bound_achieved}
