import numpy as np


def B_spline(x, n=10, k=3):
    """
        x = Array of x points,
        n = number of (equally spaced) knots
        k = spline degree
        """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    xl = np.min(x)
    xr = np.max(x)
    dx = (xr - xl) / n  # knot seperation

    t = (xl + dx * np.arange(-k, n + 1)).reshape(1, -1)
    T = np.ones_like(x).T * t  # knot matrix
    X = x.T * np.ones_like(t)  # x matrix
    P = (X - T) / dx  # seperation in natural units
    B = ((T <= X) & (X < T + dx)).astype(int)  # knot adjacency matrix
    r = np.roll(np.arange(0, t.shape[1]), -1)  # knot adjacency mask

    # compute recurrence relation k times
    for ki in range(1, k + 1):
        B = (P * B + (ki + 1 - P) * B[:, r]) / ki

    return B
