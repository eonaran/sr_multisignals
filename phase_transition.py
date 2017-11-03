"""Tools for assessing the recoverability phase transition."""

import dual_certificates
import functools
import numpy as np


def point_probability(support_fn, sign_pattern_fn, kernel, interpolation_fn,
                      num_experiments=10):
    """Estimates success probability for a single problem distribution."""
    num_successes = 0
    for _ in range(num_experiments):
        support = support_fn()
        if support is None:
            continue
        sign_pattern = sign_pattern_fn()
        interpolator = interpolation_fn(support, sign_pattern, kernel)
        success = dual_certificates.validate(
            support, sign_pattern, interpolator)['status']
        if success:
            num_successes += 1
    return num_successes / float(num_experiments)


def grid_probabilities(support_fn, sign_pattern_fn, kernel, interpolation_fn,
                       num_support_points_grid, minimum_separation_grid,
                       num_experiments=10):
    results = np.zeros(
        (len(num_support_points_grid), len(minimum_separation_grid)))
    for i, num_support_points in enumerate(num_support_points_grid):
        for j, min_separation in enumerate(minimum_separation_grid):
            this_support_fn = functools.partial(
                support_fn, num_support_points, min_separation=min_separation)
            this_sign_pattern_fn = functools.partial(
                sign_pattern_fn, num_support_points)
            results[i, j] = point_probability(
                this_support_fn,
                this_sign_pattern_fn,
                kernel,
                interpolation_fn,
                num_experiments=num_experiments)
    return results
