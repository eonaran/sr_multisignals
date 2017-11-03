import numpy as np


def _complex_gaussian():
    x, y = np.random.normal(loc=0.0, scale=1.0, size=2)
    return x + y * 1j


def uniform_sign_pattern_1d(size):
    ret = []
    for _ in range(size):
        z = _complex_gaussian()
        ret.append(z / np.linalg.norm(z))
    return np.array(ret)


def uniform_sign_pattern_multidim(size1, size2, orthogonal=False):
    if orthogonal:
        assert False
    else:
        ret = np.zeros((size1, size2)).astype(np.complex128)
        for i in range(size1):
            for j in range(size2):
                ret[i, j] = _complex_gaussian()
        norms = np.linalg.norm(ret, axis=1)
        return (ret.T / norms).T


def uniform_supports(size, min_separation=None, max_iters=10000):
    good_result = False
    iters = 0
    result = None
    while not good_result and iters < max_iters:
        result = np.random.random(size=size)
        sorted_result = np.array(sorted(result))
        this_min_separation = min(
            np.min(sorted_result[1:] - sorted_result[:-1]),
            1.0 + sorted_result[0] - sorted_result[-1])
        if (min_separation is None or
                this_min_separation >= min_separation):
            good_result = True
            break
        iters += 1

    if good_result:
        return result
    else:
        return None
