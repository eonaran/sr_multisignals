"""Tools for creating interpolation-based dual certificates."""

from trig_poly import TrigPoly, MultiTrigPoly

import mpmath
import numpy as np
from scipy import linalg as sp_linalg


def _interpolator_norm_quadratic_form(support, kernel):
    """Quadratic form calculating L2 norm of interpolator from coefficients."""
    n = support.shape[0]

    kernel_1 = kernel.derivative()

    kernel_inners = kernel.inners_of_shifts(support)
    kernel_1_inners = kernel_1.inners_of_shifts(support)
    cross_inners = kernel.inners_of_shifts_and_derivative_shifts(support)

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


def _interpolator_linear_constraints(support, sign_pattern, kernel):
    """Build linear constraint data for tangent plane derivative problem."""
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

    return problem_mx, problem_obj


def _interpolator_linear_constraints_kernel_only(
        support, sign_pattern, kernel):
    """Build linear constraint data for tangent plane derivative problem."""
    n = support.shape[0]
    m = sign_pattern.shape[1]

    time_deltas = np.outer(support, np.ones(n)) - np.outer(np.ones(n), support)

    kernel_values = kernel(time_deltas)

    sign_pattern_real = np.real(sign_pattern)
    sign_pattern_imag = np.imag(sign_pattern)

    zeros = np.zeros((n, n))
    problem_mx_rows = []
    problem_obj_cols = []
    for k in range(m):
        # Row of real part constraint
        row1 = []
        for _ in range(2 * k):
            row1.append(zeros)
        row1.append(zeros)
        row1.append(kernel_values)
        for _ in range(2 * (m - 1 - k)):
            row1.append(zeros)
        problem_mx_rows.append(row1)

        # Row of imaginary part constraint
        row2 = []
        for _ in range(2 * k):
            row2.append(zeros)
        row2.append(zeros)
        row2.append(kernel_values)
        for _ in range(2 * (m - 1 - k)):
            row2.append(zeros)
        problem_mx_rows.append(row2)

    for k in range(m):
        single_sign_pattern_real = sign_pattern_real[:, k]
        single_sign_pattern_imag = sign_pattern_imag[:, k]

        # Objective
        problem_obj_cols.append(single_sign_pattern_real)
        problem_obj_cols.append(single_sign_pattern_imag)

    problem_mx = np.bmat(problem_mx_rows)

    problem_obj_cols.append(np.zeros(n))
    problem_obj = np.hstack(problem_obj_cols)

    return problem_mx, problem_obj


def _interpolator_linear_constraints_with_derivatives(
        support, sign_pattern, kernel, support_derivatives):
    """Build linear constraint data for fixed derivative problem.

    kernel (fn)
    support (np.array(s))
    sign_pattern (np.array(s, m))
    support_derivatives (np.array(s, m))
    """
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

    problem_mx = np.bmat([
        [kernel_values, kernel1_values],
        [kernel1_values, kernel2_values]])

    coeffss = []
    for k in range(m):
        single_sign_pattern = sign_pattern[:, k]
        problem_obj = np.hstack(
            [single_sign_pattern, np.zeros(single_sign_pattern.shape[0])])
        coeffss.append(np.linalg.solve(problem_mx, problem_obj))

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
        problem_obj_cols.append(sign_pattern_real[:, k])

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
        problem_obj_cols.append(sign_pattern_imag[:, k])

    multiplier = 500

    for k in range(m):
        # Row of derivative real part constraint
        row1 = []
        for _ in range(4 * k):
            row1.append(zeros)
        row1.append(kernel_1_values / multiplier)
        row1.append(kernel_2_values / multiplier)
        row1.append(zeros)
        row1.append(zeros)
        for _ in range(4 * (m - 1 - k)):
            row1.append(zeros)
        problem_mx_rows.append(row1)
        problem_obj_cols.append(np.real(support_derivatives[:, k]) / multiplier)

        # Row of derivative imaginary part constraint
        row2 = []
        for _ in range(4 * k):
            row2.append(zeros)
        row2.append(zeros)
        row2.append(zeros)
        row2.append(kernel_1_values / multiplier)
        row2.append(kernel_2_values / multiplier)
        for _ in range(4 * (m - 1 - k)):
            row2.append(zeros)
        problem_mx_rows.append(row2)
        problem_obj_cols.append(np.imag(support_derivatives[:, k]) / multiplier)

    problem_mx = np.bmat(problem_mx_rows)
    problem_obj = np.hstack(problem_obj_cols)

    return problem_mx, problem_obj


def _optimize_quadratic_form(S, A, y, multiplier=1.0):
    """Maximize x^T S x subject to Ax = y.

    The multiplier is a factor multiplied into S in formulating the linear
    problem, which can help mitigate ill-conditioned systems (resulting from
    magnitude discrepancies between S and A).
    """
    m = A.shape[0]
    n = S.shape[0]
    # This expression for the solution is derived with Lagrange multipliers,
    # the multiplier vector of the constraint Ax = y being in the last m
    # coordinates of the result, which we discard.
    return np.linalg.solve(
        np.bmat([[multiplier * S, A.T], [A, np.zeros((m, m))]]),
        np.hstack([np.zeros(n), y]))[:n]


#
# Interpolation functions
#


def interpolate(support, sign_pattern, kernel):
    assert support.shape == sign_pattern.shape
    # assert np.all(np.absolute(np.absolute(sign_pattern) - 1.0) < 1e-10)

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
    S = _interpolator_norm_quadratic_form(support, kernel)

    coeffs = _optimize_quadratic_form(S, A, y)

    return (
        kernel.sum_shifts(-support, coeffs[:n]) +
        kernel_1.sum_shifts(-support, coeffs[n:2*n]) +
        kernel.sum_shifts(-support, coeffs[2*n:3*n] * 1j) +
        kernel_1.sum_shifts(-support, coeffs[3*n:] * 1j))


def interpolate_direct(support, sign_pattern, kernel):
    """Toy interpolation model multiplying kernel copies by sign pattern."""
    m = sign_pattern.shape[1]
    return MultiTrigPoly([
        sum(
            (kernel.shift(-t) * sign
             for t, sign in zip(support, sign_pattern[:, i])),
            TrigPoly.zero())
        for i in range(m)
    ])


def interpolate_multidim_fixed_derivatives(
        support, sign_pattern, kernel, support_derivatives,
        return_coeffs=False):
    assert support.shape[0] == sign_pattern.shape[0]
    assert np.all(
        np.absolute(
            np.sum(np.absolute(sign_pattern) ** 2, axis=1) - 1.0) < 1e-10)
    assert support_derivatives.shape == sign_pattern.shape

    n = support.shape[0]
    m = sign_pattern.shape[1]

    time_deltas = np.outer(support, np.ones(n)) - np.outer(np.ones(n), support)

    kernel_1 = kernel.derivative()
    kernel_2 = kernel_1.derivative()

    kernel_values = kernel(time_deltas)
    kernel_1_values = kernel_1(time_deltas)
    kernel_2_values = kernel_2(time_deltas)

    multiplier = 100.0
    problem_mx = np.bmat([
        [kernel_values, kernel_1_values / multiplier],
        [kernel_1_values / multiplier, kernel_2_values / multiplier / multiplier]]).real

    coeffss = []
    for k in range(m):
        problem_obj = np.hstack(
            [sign_pattern[:, k], support_derivatives[:, k] / multiplier])
        coeffss.append(np.linalg.lstsq(problem_mx, problem_obj)[0])

    return MultiTrigPoly([
        kernel.sum_shifts(-support, coeffs[:n]) +
        kernel_1.sum_shifts(-support, coeffs[n:] / multiplier)
        for coeffs in coeffss])

    problem_mx, problem_obj = (
        _interpolator_linear_constraints_with_derivatives(
            support, sign_pattern, kernel, support_derivatives))

    coeffs = np.linalg.solve(problem_mx, problem_obj)

    kernel_coeffs = [
        coeffs[4*k*n:(4*k+1)*n] + coeffs[(4*k+2)*n:(4*k+3)*n] * 1j
        for k in range(m)]
    kernel_derivative_coeffs = [
        coeffs[(4*k+1)*n:(4*k+2)*n] + coeffs[(4*k+3)*n:(4*k+4)*n] * 1j
        for k in range(m)]

    if return_coeffs:
        return kernel_coeffs, kernel_derivative_coeffs
    else:
        return MultiTrigPoly([
            kernel.sum_shifts(-support, kernel_coeffs[k]) +
            kernel.derivative().sum_shifts(
                -support, kernel_derivative_coeffs[k])
            for k in range(m)])


def interpolate_multidim_only_kernel(support, sign_pattern, kernel):
    """Interpolate only using kernels, no kernel derivatives or derivative
    constraints.
    """
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


def interpolate_multidim_l2_min(
        support, sign_pattern, kernel, return_coeffs=False):
    """Interpolation fixing derivatives at interpolated points in tangent
    plane of sphere, and minimizing L2 norm of the polynomial subject to this
    constraint.
    """
    assert support.shape[0] == sign_pattern.shape[0]
    assert np.all(
        np.absolute(
            np.sum(np.absolute(sign_pattern) ** 2, axis=1) - 1.0) < 1e-10)

    n = support.shape[0]
    m = sign_pattern.shape[1]

    problem_mx, problem_obj = _interpolator_linear_constraints(
        support, sign_pattern, kernel)

    #
    # Build objective quadratic form
    #

    # S is block-diagonal, with k blocks of size 4n x 4n each of which is the
    # same as the objective quadratic form from the one-sample case.
    S_diag_block = _interpolator_norm_quadratic_form(support, kernel)
    S = np.kron(np.identity(m), S_diag_block)

    # This multiplier value is heuristically chosen.
    multiplier = (
        kernel.derivative().squared_norm() / kernel.squared_norm() * 1e6)
    coeffs = _optimize_quadratic_form(
        S,
        problem_mx,
        problem_obj,
        multiplier=multiplier)

    kernel_coeffs = [
        coeffs[4*k*n:(4*k+1)*n] + coeffs[(4*k+2)*n:(4*k+3)*n] * 1j
        for k in range(m)]
    kernel_derivative_coeffs = [
        coeffs[(4*k+1)*n:(4*k+2)*n] + coeffs[(4*k+3)*n:(4*k+4)*n] * 1j
        for k in range(m)]

    if return_coeffs:
        return kernel_coeffs, kernel_derivative_coeffs
    else:
        return MultiTrigPoly([
            kernel.sum_shifts(-support, kernel_coeffs[k]) +
            kernel.derivative().sum_shifts(
                -support, kernel_derivative_coeffs[k])
            for k in range(m)])


def interpolate_multidim_only_kernel_l2_min(
        support, sign_pattern, kernel, return_coeffs=False):
    """Interpolation fixing derivatives at interpolated points in tangent
    plane of sphere, and minimizing L2 norm of the polynomial subject to this
    constraint.
    """
    assert support.shape[0] == sign_pattern.shape[0]
    assert np.all(
        np.absolute(
            np.sum(np.absolute(sign_pattern) ** 2, axis=1) - 1.0) < 1e-10)

    n = support.shape[0]
    m = sign_pattern.shape[1]

    problem_mx, problem_obj = _interpolator_linear_constraints_kernel_only(
        support, sign_pattern, kernel)

    print problem_mx.shape

    #
    # Build objective quadratic form
    #

    # S is block-diagonal, with k blocks of size 4n x 4n each of which is the
    # same as the objective quadratic form from the one-sample case.
    S_diag_block = kernel.inners_of_shifts(support)
    S = np.kron(np.identity(2 * m), S_diag_block)

    # This multiplier value is heuristically chosen.
    multiplier = (
        kernel.derivative().squared_norm() / kernel.squared_norm() * 1e6)
    coeffs = _optimize_quadratic_form(
        S,
        problem_mx,
        problem_obj,
        multiplier=multiplier)

    print coeffs

    kernel_coeffs = [
        coeffs[2*k*n:2*(k+1)*n] + coeffs[(k+2)*n:(2*k+3)*n] * 1j
        for k in range(m)]
    kernel_derivative_coeffs = [
        coeffs[(4*k+1)*n:(4*k+2)*n] + coeffs[(4*k+3)*n:(4*k+4)*n] * 1j
        for k in range(m)]

    if return_coeffs:
        return kernel_coeffs, kernel_derivative_coeffs
    else:
        return MultiTrigPoly([
            kernel.sum_shifts(-support, kernel_coeffs[k]) +
            kernel.derivative().sum_shifts(
                -support, kernel_derivative_coeffs[k])
            for k in range(m)])


#Fix the derivative to the difference of next and previous samples projected on the tangent plane
def interpolate_multidim_adjacent_samples(
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

    rand_scale = 1.0 / np.sqrt(n)

    vj_p1_real = np.append(np.asarray([sign_pattern_real[i + 1,:] for i in range(n-1)]), np.zeros((1,m))).reshape((n,m))
    vj_m1_real = np.append(np.zeros((1,m)), np.asarray([sign_pattern_real[i ,:] for i in range(n-1)])).reshape((n,m))
    # vj_p1_real = np.random.normal(loc=0.0, scale=rand_scale, size=(n, m))
    # vj_m1_real = np.random.normal(loc=0.0, scale=rand_scale, size=(n, m))
    vj_coeffs_real = np.diagonal((vj_p1_real - vj_m1_real).dot(sign_pattern_real.T) )

    vj_p1_imag = np.append(np.asarray([sign_pattern_imag[i + 1,:] for i in range(n-1)]), np.zeros((1,m))).reshape((n,m))
    vj_m1_imag = np.append(np.zeros((1,m)), np.asarray([sign_pattern_imag[i ,:] for i in range(n-1)])).reshape((n,m))
    # vj_p1_imag = np.random.normal(loc=0.0, scale=rand_scale, size=(n, m))
    # vj_m1_imag = np.random.normal(loc=0.0, scale=rand_scale, size=(n, m))
    vj_coeffs_imag = np.diagonal((vj_p1_imag - vj_m1_imag).dot(sign_pattern_imag.T) )

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

        # Objective
        problem_obj_cols.append(sign_pattern_real[:, k])
        problem_obj_cols.append(sign_pattern_imag[:, k])

    for k in range(m):
        row1 = []
        for _ in range(4 * k):
            row1.append(zeros)
        row1.append(kernel_1_values)
        row1.append(kernel_2_values)
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
        row2.append(kernel_1_values)
        row2.append(kernel_2_values)
        for _ in range(4 * (m - 1 - k)):
            row2.append(zeros)
        problem_mx_rows.append(row2)

        # Objective
        problem_obj_cols.append(
            (vj_p1_real[:, k] - vj_m1_real[:, k] -
             np.multiply(vj_coeffs_real, sign_pattern_real[:, k])) / 1000.0)
        problem_obj_cols.append(
            (vj_p1_imag[:, k] - vj_m1_imag[:, k] -
             np.multiply(vj_coeffs_imag, sign_pattern_imag[:, k])) / 1000.0)


    problem_mx = np.bmat(problem_mx_rows)
    problem_obj = np.hstack(problem_obj_cols)

    #
    # Solve
    #
    coeffs = np.linalg.solve(problem_mx, problem_obj)

    #Or L2 min
#     S_diag_block = _interpolator_norm_quadratic_form(kernel, support)
#     S = np.kron(np.identity(m), S_diag_block)

#     # This multiplier value is heuristically chosen.
#     multiplier = kernel_1.squared_norm() / kernel.squared_norm() * 1e3
#     coeffs = _optimize_quadratic_form(
#         S,
#         problem_mx,
#         problem_obj,
#         multiplier=multiplier)


    return MultiTrigPoly([
        (kernel.sum_shifts(-support, coeffs[4*k*n:(4*k+1)*n]) +
         kernel_1.sum_shifts(-support, coeffs[(4*k+1)*n:(4*k+2)*n]) +
         kernel.sum_shifts(-support, coeffs[(4*k+2)*n:(4*k+3)*n] * 1j) +
         kernel_1.sum_shifts(-support, coeffs[(4*k+3)*n:(4*k+4)*n] * 1j))
        for k in range(m)])


def interpolate_multidim_0Grad(support, sign_pattern, kernel):
    assert support.shape[0] == sign_pattern.shape[0]
    assert np.all(
        np.absolute(
            np.sum(np.absolute(sign_pattern) ** 2, axis=1) - 1.0) < 1e-10)

    n = support.shape[0]
    m = sign_pattern.shape[1]

    kernel1 = kernel.derivative()
    kernel2 = kernel1.derivative()

    time_deltas = np.outer(support, np.ones(n)) - np.outer(np.ones(n), support)
    kernel_values = kernel(time_deltas)
    kernel1_values = kernel1(time_deltas)
    kernel2_values = kernel2(time_deltas)
    problem_mx = np.bmat([
        [kernel_values, kernel1_values],
        [kernel1_values, kernel2_values]])

    coeffss = []
    for k in range(m):
        single_sign_pattern = sign_pattern[:, k]
        problem_obj = np.hstack(
            [single_sign_pattern, np.zeros(single_sign_pattern.shape[0])])
        coeffss.append(np.linalg.solve(problem_mx, problem_obj))

    return MultiTrigPoly([
        (TrigPoly(
            kernel.freqs,
            sum(kernel.coeffs * np.exp(2.0 * np.pi * 1j * kernel.freqs * -t) * c
                for c, t in zip(coeffs[:n], support))) +
         TrigPoly(
             kernel1.freqs,
             sum(kernel1.coeffs * np.exp(2.0 * np.pi * 1j * kernel1.freqs * -t) * c
                 for c, t in zip(coeffs[n:], support))))
        for coeffs in coeffss])

    return MultiTrigPoly([
        sum([kernel.shift(-t) * c for c, t in zip(coeffs[:n], support)],
            TrigPoly.zero()) +
        sum([kernel1.shift(-t) * c for c, t in zip(coeffs[n:], support)],
            TrigPoly.zero())
        for coeffs in coeffss])

#
# Validation functions
#

_EPSILON = 1e-7

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
