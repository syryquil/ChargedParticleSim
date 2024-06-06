from functools import partial

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from psutil import virtual_memory
from tqdm import tqdm

class Grid:
    def __init__(self, x_max: float, y_max: float, n_x: int, n_y: int):
        self.x_max, self.y_max = x_max, y_max
        self.n_x, self.n_y = n_x, n_y

        self.X, self.dx = np.linspace(0, x_max, n_x, retstep=True)
        self.Y, self.dy = np.linspace(0, y_max, n_y, retstep=True)

        self.xg, self.yg = np.meshgrid(self.X, self.Y)
        self.values = None

    def apply(self, func, kill_threads=False, **kwargs):  # func must return a scalar or 1D vector. Applies function to locations in parallel
        if kill_threads or virtual_memory().percent > 95:
            pool = Pool()
            pool.clear()

        elif not kill_threads:
            name = func.func.__name__ if isinstance(func, partial) else func.__name__

            shape = np.shape(func((self.X[0], self.Y[0]), **kwargs))
            n = shape[0] if shape else None

            if shape == ():
                new_shape = (self.n_x, self.n_y)
            elif shape == (n,):
                new_shape = (self.n_x, self.n_y, n)
            else:
                raise ValueError("func must return a scalar or 1D vector")

            locs = [loc for loc in np.vstack((self.xg.flatten(), self.yg.flatten())).T]
            with Pool() as pool:
                values = []
                N = self.n_x * self.n_y
                for value in tqdm(pool.imap(func, locs, chunksize=N//(5*pool.ncpus)), total=N, leave=False, position=1, desc=f'calculating {name}'):
                    values.append(value)

                values = np.array(values).reshape(new_shape)

            self.values = values

    @staticmethod
    def kill_multithreading(): #rancid stuff going on here
        kill_grid = Grid(1, 1, 1, 1)
        kill_grid.apply(lambda x: 0, kill_threads=True)