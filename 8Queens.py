from itertools import permutations
import time
import numpy as np

as_strided = np.lib.stride_tricks.as_strided


def diagonals_with_trace(x):
    # sum_of_diagonals returns a one-dimensional array with sum of all diagonals
    # if the sum > 2 it means that more than 1 Queen is placed on this diagonal
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
    itemsize = x.itemsize
    sums = []
    sums += [as_strided(x[0, i:], shape=(n-i,), strides=((n+1)*itemsize,)).sum() for i in range(n)]
    sums += [as_strided(x[0, -(i+1):], shape=(n-i,), strides=((n-1)*itemsize,)).sum() for i in range(n)]
    sums += [as_strided(x[i:, 0], shape=(n-i,), strides=((n+1)*itemsize,)).sum() for i in range(1, n)]
    sums += [as_strided(x[i:, -1], shape=(n-i,), strides=((n-1)*itemsize,)).sum() for i in range(1, n)]
    return np.array(sums)


def fancy_indexing(x):
    rows, cols = x.shape
    jj = np.tile(np.arange(cols), rows)
    ii = (np.arange(cols)+np.arange(rows)[::-1, None]).ravel()
    # Temporary array of 0s
    z = np.zeros(((rows+cols-1)*2, cols), int)
    # Assign diagonals to rows in one direction
    z[ii, jj] = x.ravel()
    # The other direction
    z[ii+(rows+cols-1), jj] = x[::-1, ::].ravel()
    return z.sum(axis=1)


def stack_n_stride(x):
    rows, cols = x.shape
    if cols > rows:
        x = x.T
        rows, cols = x.shape
    fill = np.zeros((cols - 1, cols), dtype=x.dtype)
    stacked = np.vstack((x, fill, x[::, ::-1], fill, x))
    major_stride, minor_stride = stacked.strides
    strides = major_stride, minor_stride * (cols + 1)
    shape = ((rows + cols - 1)*2, cols)
    return as_strided(stacked, shape, strides).sum(1)


def binc(x):
    rows, cols = x.shape
    indices = (np.arange(rows) + np.arange(cols)[::1, None]).ravel()
    return np.concatenate((np.bincount(indices, weights=x[::, ::-1].ravel()),
                           np.bincount(indices, weights=x.ravel())))


a = np.eye(8)
# a = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0, 0, 0, 0],
#               [0, 0, 1, 0, 0, 0, 0, 0],
#               [0, 0, 0, 1, 0, 0, 0, 0],
#               [0, 0, 0, 0, 1, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 0, 0, 1]])

sum_of_diagonals = [diagonals_with_trace, diagonals_with_strides, fancy_indexing, stack_n_stride, binc]

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
