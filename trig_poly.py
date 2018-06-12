import operator
import numbers
import numpy as np


class TrigPoly(object):

    def __init__(self, freqs, coeffs):
        assert len(freqs) == len(coeffs)
        self.freqs_hash = hash(tuple(freqs))
        self.freqs = np.array(freqs)
        self.coeffs = np.array(coeffs)
        self.coeff_dict = {f: c for f, c in zip(freqs, coeffs)}

    @classmethod
    def dirichlet(cls, f):
        freqs = range(-f, f + 1)
        coeffs = [1.0 / (2.0 * f + 1.0) for _ in range(len(freqs))]
        return cls(freqs, coeffs)

    @classmethod
    def multi_dirichlet(cls, f, gammas):
        return reduce(
            operator.mul,
            [cls.dirichlet(int(f * gamma)) for gamma in gammas],
            cls.one())

    @classmethod
    def zero(cls):
        return cls([0], [0.0])

    @classmethod
    def one(cls):
        return cls([0], [1.0])

    def eval(self, t):
        if isinstance(t, numbers.Number):
            t = np.array([t])
        repeated_ts = np.repeat(
            t.reshape([1] + list(t.shape)), len(self.freqs), axis=0)
        reshaped_freqs = (
            self.freqs.reshape([len(self.freqs)] + [1] * len(t.shape)))
        reshaped_coeffs = (
            self.coeffs.reshape([len(self.coeffs)] + [1] * len(t.shape)))

        return np.sum(
            reshaped_coeffs *
            np.exp(2.0 * np.pi * 1j * repeated_ts * reshaped_freqs),
            axis=0)

    def conjugate(self):
        return TrigPoly([-f for f in self.freqs], np.conj(self.coeffs))

    def real(self):
        return (self + self.conjugate()) * 0.5

    def imag(self):
        return (self + self.conjugate() * (-1.0)) * (-0.5 * 1j)

    def shift(self, t):
        """Returns the trig poly time-shifted by +t."""
        new_coeffs = self.coeffs * np.exp(2.0 * np.pi * 1j * self.freqs * t)
        return TrigPoly(self.freqs, new_coeffs)

    def sum_shifts(self, shifts, coeffs):
        return TrigPoly(
            self.freqs,
            sum(self.coeffs * np.exp(2.0 * np.pi * 1j * self.freqs * t) * c
                for c, t in zip(coeffs, shifts)))

    def squared_norm(self):
        return np.sum(np.absolute(self.coeffs) ** 2)

    def derivative(self):
        return TrigPoly(
            self.freqs, self.coeffs * 2.0 * np.pi * 1j * self.freqs)

    def inners_of_shifts(self, ts):
        n = ts.shape[0]

        # deltas[i, j] = t_i - t_j
        deltas = (
            np.outer(ts, np.ones(n)) - np.outer(np.ones(n), ts)
            ).reshape((n, n, 1)).repeat(len(self.freqs), axis=2)

        return np.einsum(
            'ijk,k->ij',
            np.exp(2.0 * np.pi * 1j * deltas * self.freqs),
            np.absolute(self.coeffs) ** 2)

    def inners_of_shifts_and_derivative_shifts(self, ts):
        n = ts.shape[0]

        # deltas[i, j] = t_i - t_j
        deltas = (
            np.outer(ts, np.ones(n)) - np.outer(np.ones(n), ts)
            ).reshape((n, n, 1)).repeat(len(self.freqs), axis=2)

        return np.einsum(
            'ijk,k,k->ij',
            np.exp(2.0 * np.pi * 1j * deltas * self.freqs),
            2.0 * np.pi * 1j * self.freqs,
            np.absolute(self.coeffs) ** 2)

    def __call__(self, t):
        return self.eval(t)

    def __add__(self, other):
        if self.freqs_hash == other.freqs_hash:
            return TrigPoly(self.freqs, self.coeffs + other.coeffs)

        all_freqs = sorted(set(self.freqs) | set(other.freqs))
        all_coeffs = [
            self.coeff_dict.get(f, 0.0) + other.coeff_dict.get(f, 0.0)
            for f in all_freqs]
        return TrigPoly(all_freqs, all_coeffs)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            new_coeffs = [c * other for c in self.coeffs]
            return TrigPoly(self.freqs, new_coeffs)
        elif isinstance(other, TrigPoly):
            min_freq = min(min(self.freqs), min(other.freqs))
            max_freq = max(max(self.freqs), max(other.freqs))
            all_freqs = range(min_freq, max_freq + 1)

            all_self_coeffs = np.zeros(len(all_freqs), dtype=np.complex)
            for f, c in zip(self.freqs, self.coeffs):
                all_self_coeffs[f - min_freq] = c

            all_other_coeffs = np.zeros(len(all_freqs), dtype=np.complex)
            for f, c in zip(other.freqs, other.coeffs):
                all_other_coeffs[f - min_freq] = c

            new_freqs = range(2 * min_freq, 2 * max_freq + 1)
            new_coeffs = np.convolve(all_self_coeffs, all_other_coeffs)

            return TrigPoly(new_freqs, new_coeffs)
        else:
            assert False


class MultiTrigPoly(object):

    def __init__(self, polys):
        assert all(isinstance(p, TrigPoly) for p in polys)
        self.polys = polys

    def derivative(self):
        return MultiTrigPoly([p.derivative() for p in self.polys])

    def eval(self, t):
        return np.stack([p(t) for p in self.polys], axis=0)

    def __call__(self, t):
        return self.eval(t)
