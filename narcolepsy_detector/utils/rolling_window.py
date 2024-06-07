import math

import numpy as np
from skimage.util import view_as_windows


def rolling_window_nodelay(vec, window, step):
    def calculate_padding(vec, window, step):

        N = len(vec)
        B = math.ceil(N / step)
        L = (B - 1) * step + window
        return L - N

    pad = calculate_padding(vec, window, step)
    A = view_as_windows(np.pad(vec, (0, pad)), window, step).T
    zero_cols = pad // step
    return np.delete(A, np.arange(A.shape[1] - zero_cols, A.shape[1]), axis=1)


def rolling_window(a, window_size, step, axis=0):
    if axis == 0:
        shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
        strides = (a.strides[0],) + a.strides
    else:
        raise NotImplementedError
    rolled = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return rolled[np.arange(0, shape[0], step)]
