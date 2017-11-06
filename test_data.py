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
        # Generate orthogonal columns by taking the QR decomposition of a
        # random GUE matrix
        ret = np.zeros((size1, size1))
        iid_normals = (
            np.random.normal(loc=0.0, scale=1.0, size=(size1, size1)) +
            np.random.normal(loc=0.0, scale=1.0, size=(size1, size1)) * 1j) / (
                np.sqrt(2.0))
        gue = (iid_normals + iid_normals.T) / np.sqrt(2.0)
        q = np.linalg.qr(gue)[0]
        return q[:, :size2]
    else:
        ret = np.zeros((size1, size2)).astype(np.complex128)
        for i in range(size1):
            for j in range(size2):
                ret[i, j] = _complex_gaussian()
        norms = np.linalg.norm(ret, axis=1)
        return (ret.T / norms).T


class CircleInterval(object):
    """Represents an interval on the circle thought of as [0, 1] interval."""

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.wraps_around = end < start
        self.is_entire = start == 0.0 and end == 1.0

    def contains(self, x):
        if not self.wraps_around:
            return self.start <= x <= self.end
        else:
            return x <= self.end or x >= self.start

    def length(self):
        if not self.wraps_around:
            return self.end - self.start
        else:
            return 1.0 + self.end - self.start

    def random(self):
        """Samples a uniformly random point from the interval."""
        if not self.wraps_around:
            return self.start + np.random.random() * self.length()
        else:
            out = self.start + np.random.random() * self.length()
            if out >= 1.0:
                return out - 1.0
            else:
                return out

    def split_with_separation(self, x, delta):
        assert self.contains(x)

        if delta == 0.0:
            return [self]

        lo = x - delta
        hi = x + delta
        if hi >= 1.0:
            hi = hi - 1.0
        if lo <= 0.0:
            lo = lo + 1.0

        if self.is_entire:
            return [CircleInterval(hi, lo)]
        else:
            resulting_intervals = []

            if self.contains(lo):
                resulting_intervals.append(CircleInterval(self.start, lo))
            if self.contains(hi):
                resulting_intervals.append(CircleInterval(hi, self.end))

            return resulting_intervals

    def __repr__(self):
        return 'CircleInterval(%.2f, %.2f)' % (self.start, self.end)


def uniform_supports(size, min_separation=None, max_iters=1000):
    if min_separation is None:
        min_separation = 0.0

    assert size * min_separation <= 1.0

    for _ in range(max_iters):
        pts = []
        intervals = [CircleInterval(0.0, 1.0)]
        for _ in range(size):
            if not intervals:
                pts = []
                break

            # Choose a random interval, weighted by length:
            lengths = np.array([interval.length() for interval in intervals])
            probs = lengths / np.sum(lengths)
            interval_ix = np.random.choice(len(intervals), 1, p=probs)[0]
            interval = intervals[interval_ix]

            # Choose a random point in the chosen interval
            x = interval.random()

            # Add to list of support points
            pts.append(x)

            # Adjust intervals list
            resulting_intervals = interval.split_with_separation(
                x, min_separation)
            intervals.pop(interval_ix)
            intervals.extend(resulting_intervals)
        else:
            break

    if not pts:
        return None
    else:
        return np.array(sorted(pts))
