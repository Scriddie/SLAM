import numpy as np

class Grid:
    def __init__(self, size, cell_size):
        self.size = size
        self.grid = {}

    # TODO see how we can plot in pygame
    def grid_to_mat(self):
        """ convert grid to matrix """
        # mat = np.zeros(self.size)
        # for (i, j), v in self.grid.items():
        #     robo_map[i, j] = 1
        pass