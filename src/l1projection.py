import numpy as np


def projection_onto_simplex(x, simplex):
    shape, = x.shape
    assert simplex >= 0
    if x.sum() == simplex and np.alltrue(x >= 0):
        return x

    sorted_x = np.sort(x)[::-1]
    cumsum = np.cumsum(sorted_x)
    rho = np.nonzero((sorted_x * np.arange(1, shape + 1))
                     >
                     (cumsum - simplex))[0][-1] + 1
    theta = (cumsum[rho-1] - simplex)/rho
    w = (x-theta).clip(min=0)
    return w


def projection_onto_l1ball(x, s):
    assert s > 0
    absx = np.abs(x)
    if absx.sum() <= s:
        return x

    w = projection_onto_simplex(absx, s)
    w *= np.sign(x)
    return w


