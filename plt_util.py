"""Tools for plotting diagnostic info."""

from matplotlib import pyplot as plt
import numpy as np


def plot_trig_poly_magnitude(p, ax=None, points=1000, c='blue', **kwargs):
    ax = ax or plt.gca()

    ts = np.linspace(0.0, 1.0, points)
    values = p(ts)
    if len(values.shape) == 1:
        ys = np.absolute(values)
    else:
        ys = np.linalg.norm(values, axis=0)

    ax.plot(ts, ys, c=c, **kwargs)


def plot_support_magnitude_lines(support, ax=None, c='green'):
    ax = ax or plt.gca()

    ax.vlines(support, 0.0, 1.0, color=c)


def plot_magnitude_bounds(ax=None, c='red'):
    ax = ax or plt.gca()

    ax.hlines([0.0, 1.0], 0.0, 1.0, color=c)
