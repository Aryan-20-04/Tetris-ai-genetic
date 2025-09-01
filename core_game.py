# core_game.py (headless, vectorized)
import numpy as np

cols, rows = 10, 20
ROWS, COLS = rows, cols

tetrominoes = {
    'T': [
        np.array([[0,1,0],
                  [1,1,1]], dtype=np.int8),
        np.array([[1,0],
                  [1,1],
                  [1,0]], dtype=np.int8),
        np.array([[1,1,1],
                  [0,1,0]], dtype=np.int8),
        np.array([[0,1],
                  [1,1],
                  [0,1]], dtype=np.int8)
    ],
    'O': [
        np.array([[1,1],
                  [1,1]], dtype=np.int8)
    ],
    'I': [
        np.array([[1,1,1,1]], dtype=np.int8),
        np.array([[1],
                  [1],
                  [1],
                  [1]], dtype=np.int8)
    ],
    'S': [
        np.array([[0,1,1],
                  [1,1,0]], dtype=np.int8),
        np.array([[1,0],
                  [1,1],
                  [0,1]], dtype=np.int8)
    ],
    'Z': [
        np.array([[1,1,0],
                  [0,1,1]], dtype=np.int8),
        np.array([[0,1],
                  [1,1],
                  [1,0]], dtype=np.int8)
    ],
    'J': [
        np.array([[1,0,0],
                  [1,1,1]], dtype=np.int8),
        np.array([[1,1],
                  [1,0],
                  [1,0]], dtype=np.int8),
        np.array([[1,1,1],
                  [0,0,1]], dtype=np.int8),
        np.array([[0,1],
                  [0,1],
                  [1,1]], dtype=np.int8)
    ],
    'L': [
        np.array([[0,0,1],
                  [1,1,1]], dtype=np.int8),
        np.array([[1,0],
                  [1,0],
                  [1,1]], dtype=np.int8),
        np.array([[1,1,1],
                  [1,0,0]], dtype=np.int8),
        np.array([[1,1],
                  [0,1],
                  [0,1]], dtype=np.int8)
    ]
}

def create_grid():
    return np.zeros((rows, cols), dtype=np.int8)

def check_collision(grid, shape, x, y):
    h, w = shape.shape
    if x < 0 or x + w > cols or y + h > rows:
        return True
    region = grid[y:y+h, x:x+w]
    return np.any((region != 0) & (shape != 0))

def lock_piece(grid, shape, x, y, color=1):
    h, w = shape.shape
    region = grid[y:y+h, x:x+w]
    mask = shape != 0
    region[mask] = color
    grid[y:y+h, x:x+w] = region
    return grid

def clear_lines(grid):
    full_rows = np.where(np.all(grid != 0, axis=1))[0]
    lines_cleared = len(full_rows)
    if lines_cleared > 0:
        grid = np.delete(grid, full_rows, axis=0)
        new_rows = np.zeros((lines_cleared, grid.shape[1]), dtype=np.int8)
        grid = np.vstack((new_rows, grid))
    return grid, lines_cleared

def get_drop_y(grid: np.ndarray, shape: np.ndarray, x: int) -> int:
    """
    Compute the lowest valid y-position for placing shape at column x.
    Returns -1 if placement is invalid.
    """
    h, w = shape.shape
    max_y = grid.shape[0] - h

    # Out of bounds
    if x < 0 or x + w > grid.shape[1]:
        return -1

    for y in range(max_y + 1):
        region = grid[y:y+h, x:x+w]
        if np.any((region != 0) & (shape != 0)):
            return y - 1
    return max_y


def place_piece(grid: np.ndarray, shape: np.ndarray, x: int, y: int, color: int) -> np.ndarray:
    """
    Place shape on a copy of grid at (x, y) with given color.
    """
    new_grid = grid.copy()
    h, w = shape.shape
    mask = shape != 0
    new_grid[y:y+h, x:x+w][mask] = color
    return new_grid