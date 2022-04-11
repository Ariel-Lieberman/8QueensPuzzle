from itertools import permutations
import time
import numpy as np

as_strided = np.lib.stride_tricks.as_strided


def all_diagonals(x):
    n = x.shape[0]
    sums = []
    sums += [x.trace(i) for i in range(n)]
    sums += [x.trace(-i) for i in range(1, n)]
    rot = np.rot90(x, 3)
    sums += [rot.trace(i) for i in range(n)]
    sums += [rot.trace(-i) for i in range(1, n)]
    return np.array(sums)


def diagonals_with_strides(x):
    n = x.shape[0]
    sums = []
    sums += [as_strided(x[0, i:], shape=(n-i,), strides=((n+1)*x.itemsize,)).sum() for i in range(n)]
    sums += [as_strided(x[0, -(i+1):], shape=(n-i,), strides=((n-1)*x.itemsize,)).sum() for i in range(n)]
    sums += [as_strided(x[i:, 0], shape=(n-i,), strides=((n+1)*x.itemsize,)).sum() for i in range(1, n)]
    sums += [as_strided(x[i:, -1], shape=(n-i,), strides=((n-1)*x.itemsize,)).sum() for i in range(1, n)]
    return np.array(sums)


def broadcasting(x):
    jj = np.tile(np.arange(x.shape[1]), x.shape[0])
    ii = (np.arange(x.shape[1])+np.arange(x.shape[0])[::-1, None]).ravel()
    z = np.zeros(((x.shape[0]+x.shape[1]-1)*2, x.shape[1]), int)
    z[ii, jj] = x.ravel()
    # The other direction
    z[ii+(x.shape[0]+x.shape[1]-1), jj] = np.flipud(x).ravel()
    return z.sum(axis=1)


def stack_n_stride(x):
    rows, cols = x.shape
    if cols > rows:
        x = x.T
        rows, cols = x.shape
    fill = np.zeros((cols - 1, cols), dtype=x.dtype)
    stacked = np.vstack((x, fill, np.fliplr(x), fill, x))
    major_stride, minor_stride = stacked.strides
    strides = major_stride, minor_stride * (cols + 1)
    shape = ((rows + cols - 1)*2, cols)
    return as_strided(stacked, shape, strides).sum(1)


def binc(x):
    indices = (np.arange(x.shape[1]) + np.arange(x.shape[0])[::1, None]).ravel()
    return np.concatenate((np.bincount(indices, weights=np.fliplr(x).ravel()),
                           np.bincount(indices, weights=x.ravel())))


a = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]])


sum_of_diagonals = [all_diagonals, diagonals_with_strides, broadcasting, stack_n_stride, binc]

Iter = 5

for f in sum_of_diagonals:
    s = 0
    t = time.time()
    for _ in range(Iter):
        for p in permutations(a):
            sd = f(np.array(p, dtype=np.byte))
            if (sd < 2).all():
                s += 1
                # print(i, ':\n', np.array(p))
    print(f"{f.__name__} ({s / Iter}): {(time.time() - t) / Iter}")
