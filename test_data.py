import numpy as np


def uniform_sign_pattern_1d(size):
    ret = []
    for _ in range(size):
        x, y = np.random.normal(loc=0.0, scale=1.0, size=2)
        norm = np.sqrt(x ** 2 + y ** 2)
        x, y = x / norm, y / norm
        ret.append(x + y * 1j)
    return np.array(ret)


def uniform_supports(size, min_separation=None, max_iters=10000):
    good_result = False
    iters = 0
    result = None
    while not good_result and iters < max_iters:
        result = np.random.random(size=size)
        sorted_result = np.array(sorted(result))
        if (min_separation is None or
                np.min(sorted_result[1:] - sorted_result[:-1]) >=
                min_separation):
            good_result = True
            break
        iters += 1

    if good_result:
        return result
    else:
        return None
