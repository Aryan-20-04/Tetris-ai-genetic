# adapter.py (optimized)
import random
import numpy as np
import core_game as cg  # your vectorized module

class CoreGameAdapter:
    # expose the attributes AIAgent/GA expect
    cols = cg.cols
    rows = cg.rows

    def __init__(self):
        # Pre-allocate commonly used shapes for performance
        self._shape_cache = {}
        
    def create_grid(self):
        return cg.create_grid()

    # keep legacy method names used by AIAgent/GA
    def checkCollision(self, grid, shape, x, y):
        # Convert and cache shape if needed
        if not isinstance(shape, np.ndarray):
            shape = np.array(shape, dtype=np.int8)
        return cg.check_collision(grid, shape, x, y)

    def lockTetromino(self, grid, shape, x, y, color=1):
        # for training we keep the grid binary: color=1
        if not isinstance(shape, np.ndarray):
            shape = np.array(shape, dtype=np.int8)
        return cg.lock_piece(grid, shape, x, y, color)

    def clear_lines(self, grid):
        return cg.clear_lines(grid)

    def new_tetromino(self):
        # returns: (list_of_rotations, rotation_idx, color)
        # color is irrelevant in headless training; use 1
        k = random.choice(list(cg.tetrominoes.keys()))
        rotations = cg.tetrominoes[k]
        return rotations, 0, 1  # (shape_list, rotation, color)