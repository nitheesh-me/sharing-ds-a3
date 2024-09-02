"""Parallel Matrix Chain Multiplication Problem

run with:
mpiexec -n 11 python 2/julia_set.py ./2/input.txt
"""
import math
import logging
from itertools import product
import mpi4py
mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False    # do not finalize MPI automatically
from mpi4py import MPI # import the 'MPI' module
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def _load_input():
    import sys
    with open(sys.argv[1], 'r') as f:
        data = f.readlines()
    N, M, k = list(map(int, data[0].strip().split()))
    complex_number = complex(*list(map(float, data[1].strip().split())))
    return N, M, k, complex_number


def split_grid(N, M, k):
    # Calculate the number of splits in rows and columns
    if N > M:
        rows_split = k
        cols_split = 1
    else:
        cols_split = k
        rows_split = 1

    # Calculate the ranges of rows and columns for each split
    ranges = []
    for row in range(rows_split):
        for col in range(cols_split):
            row_range = (row * N // rows_split, (row + 1) * N // rows_split)
            col_range = (col * M // cols_split, (col + 1) * M // cols_split)
            ranges.append((*row_range, *col_range))
    return ranges, rows_split, cols_split


def is_julia(z, c, k, T):
    """
    check if a pixel is in the julia set or not
    """
    for i in range(k):
        # z = z * z + c
        _xtemp = z.real * z.real - z.imag * z.imag + c.real
        _ytemp = 2 * z.real * z.imag + c.imag
        z = complex(_xtemp, _ytemp)
        if abs(z) > T:
            return False
    return True


def julia_set(x_coordinate, y_coordinate, c, k, T):
    """
    compute the julia set for a given rectangle
    """
    arr = []
    for i in range(len(x_coordinate)):
        arr.append([0] * len(y_coordinate))

    for i, x in enumerate(x_coordinate):
        for j, y in enumerate(y_coordinate):
            z = complex(x, y)
            if is_julia(z, c, k, T):
                arr[i][j] = 1
            else:
                arr[i][j] = 0
    return arr


def julia_set_vectorized(x_coordinate, y_coordinate, c, k, T):
    """
    compute the julia set for a given rectangle
    """
    f = lambda z: is_julia(z, c, k, T)
    arr = np.vectorize(f)(np.array([[complex(y, x) for y in y_coordinate] for x in x_coordinate]))
    return arr.tolist()


def _generate_points(N, M, LIMITS, index=1):
    """
    Generate points on the plane
    """
    x_coordinate = [i * (LIMITS[1] - LIMITS[0]) / N + LIMITS[0] for i in range(index, N+index)]
    y_coordinate = [j * (LIMITS[3] - LIMITS[2]) / M + LIMITS[2] for j in range(index, M+index)]
    return x_coordinate, y_coordinate


def master_node(Comm):
    N, M, k, complex_number = _load_input()
    LIMITS = (-1.5, 1.5, -1.5, 1.5)
    T = 2
    size = Comm.Get_size()
    logger.info(f'We have {size} processes to do the work for {N} matrices')
    split_size = math.ceil(N / size)
    logger.info(f'We will split the work into {split_size} matrices each')
    for i in range(size):
        Comm.send(k, dest=i, tag=1)
        Comm.send(complex_number, dest=i, tag=2)
        Comm.send(T, dest=i, tag=3)

    # send reduced N, M, LIMITS to all threads
    # dimensions use size
    ranges, rows_split, cols_split = split_grid(N, M, size)
    x_coordinate, y_coordinate = _generate_points(N, M, LIMITS)

    logger.info(ranges)
    for i, (r_start, r_end, c_start, c_end) in enumerate(ranges):
        # send the coordinates to all threads
        Comm.send(x_coordinate[r_start:r_end], dest=i, tag=4)
        Comm.send(y_coordinate[c_start:c_end], dest=i, tag=5)

    compute_node(Comm)  # Apply the work to all threads, including master node

    # receive all results from threads non-blocking
    results = np.zeros((N, M), dtype=int)
    for i in range(size):
        logger.info(ranges[i])
        results[ranges[i][0]:ranges[i][1], ranges[i][2]:ranges[i][3]] = Comm.recv(source=i, tag=6)

    for row in results:
        print(*row)
    logger.info(results[16,4])
    logger.info(results[10,6])



def compute_node(Comm):
    N, M, K, c, T = None, None, None, None, None
    K = Comm.recv(source=0, tag=1)
    c = Comm.recv(source=0, tag=2)
    T = Comm.recv(source=0, tag=3)
    logger.info(f'rank {Comm.rank} have k={K}, c={c}, T={T}')
    x_coordinate = Comm.recv(source=0, tag=4)
    y_coordinate = Comm.recv(source=0, tag=5)
    N = len(x_coordinate)
    M = len(y_coordinate)
    logger.info(f'rank {Comm.rank} have {N} rows and {M} columns')

    # get the julia set for each matrix
    # js = julia_set(x_coordinate, y_coordinate, c, K, T)
    js = julia_set_vectorized(x_coordinate, y_coordinate, c, K, T)
    logger.info(f"Computed julia set at {Comm.rank}")
    # send the julia set to all threads
    Comm.isend(js, dest=0, tag=6)

if __name__ == '__main__':
    logger.info(f"Vendor: {"".join(MPI.get_vendor())}")
    Comm = MPI.COMM_WORLD

    MPI.Init()      # manual initialization of the MPI environment
    # --------------------------------------------------------------------------
    if Comm.rank == 0:
        logger.info(f"I am the master node on {Comm.name}")
        master_node(Comm)
    else:
        logger.info(f"Hi, I am rank {Comm.rank + 1} out of {Comm.size} processes on {Comm.name}.")
        compute_node(Comm)
    # --------------------------------------------------------------------------
    MPI.Finalize()  # manual finalization of the MPI environment
