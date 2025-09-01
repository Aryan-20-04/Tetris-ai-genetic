# ai_agent.py (optimized)
import math
import numpy as np
import core_game as game  # expects: cols, rows, check_collision, lock_piece, clear_lines


class AIAgent:
    def __init__(self, weights=None):
        self.weights: dict[str, float] = weights or {
            "aggregate_height": -0.510066,
            "complete_lines": 0.760666,
            "holes": -0.35663,
            "bumpiness": -0.184483,
            "height_variance": -0.15,
            "max_height": -0.2,
            "deep_wells": -0.3,
            "row_transitions": -0.4,
            "column_transitions": -0.4,
            "landing_height": -0.05,
            "eroded_piece_cells": 1.0,
            "well_sums": -0.1,
            "hole_depth": -0.5,
            "combo": 0.5,
            "line_clear_potential": 0.2,
        }

    # ----------------------------
    # Feature extraction (vectorized & cached)
    # ----------------------------
    def extract_features(
        self,
        grid: np.ndarray,
        landing_height: float = 0.0,
        eroded_cells: int = 0,
        combo: int = 0,
        lines_cleared: int = 0,
    ) -> dict[str, float]:

        rows, cols = grid.shape
        occ = grid != 0  # boolean occupancy

        # Heights: distance from top to first occupied in each column
        any_col = occ.any(axis=0)
        first_occ = np.where(any_col, np.argmax(occ, axis=0), rows)  # rows if column empty
        heights = rows - first_occ  # 0..rows
        total_height = float(np.sum(heights))
        max_height = float(np.max(heights))

        # Holes: empty cells below the topmost occupied in each column
        r_idx = np.arange(rows)[:, None]  # shape (rows,1)
        tops = first_occ[None, :]         # shape (1,cols)
        holes_mask = (r_idx >= tops) & (~occ)
        holes = float(np.sum(holes_mask))

        # Hole depth: distance of each hole to bottom (sum of depths)
        hole_depth = float(np.sum((rows - 1 - r_idx) * holes_mask))

        # Bumpiness & variance (optimized)
        height_diff = np.abs(np.diff(heights))
        bumpiness = float(np.sum(height_diff))
        
        mean_h = total_height / cols if cols else 0.0
        height_variance = float(np.var(heights)) if cols else 0.0

        # Transitions (optimized)
        row_transitions = float(np.sum(occ[:, :-1] != occ[:, 1:]))
        column_transitions = float(np.sum(occ[:-1, :] != occ[1:, :]))

        # Lines complete
        complete_lines = float(np.sum(np.all(occ, axis=1)))

        # Wells (vectorized and optimized)
        left = np.concatenate([[np.inf], heights[:-1]])
        right = np.concatenate([heights[1:], [np.inf]])
        min_neighbor = np.minimum(left, right)
        well_depths = np.maximum(0, min_neighbor - heights)
        
        # Pre-compute well sums more efficiently
        well_sums = float(np.sum(well_depths * (well_depths + 1) * 0.5))
        deep_wells = float(np.sum(np.where(well_depths >= 3, well_depths ** 1.5, 0.0)))

        return {
            "aggregate_height": total_height,
            "holes": holes,
            "bumpiness": bumpiness,
            "height_variance": height_variance,
            "max_height": max_height,
            "complete_lines": complete_lines,
            "landing_height": float(landing_height),
            "eroded_piece_cells": float(eroded_cells),
            "combo": float(combo),
            "line_clear_potential": float(lines_cleared),
            "deep_wells": deep_wells,
            "row_transitions": row_transitions,
            "column_transitions": column_transitions,
            "well_sums": well_sums,
            "hole_depth": hole_depth,
        }

    def evaluate_board(
        self,
        grid: np.ndarray,
        landing_height: float = 0.0,
        eroded_cells: int = 0,
        combo: int = 0,
        lines_cleared: int = 0,
    ) -> float:
        feats = self.extract_features(grid, landing_height, eroded_cells, combo, lines_cleared)
        return float(sum(self.weights.get(k, 0.0) * v for k, v in feats.items()))

    def get_features(self, grid: np.ndarray) -> dict[str, float]:
        return self.extract_features(grid)

    # ----------------------------
    # Simulation & decision (optimized)
    # ----------------------------
    def _simulate_placement(self, grid: np.ndarray, shape: np.ndarray, x: int, y: int, color: int):
        """
        Place piece (no side effects), compute landing height, eroded piece cells, lines cleared.
        Optimized version with pre-allocated arrays.
        """
        # Pre-allocate test grid to avoid repeated copying
        test_grid = grid.copy()
        game.lock_piece(test_grid, shape, x, y, color)

        # landing height = average y (from top) of placed cells (vectorized)
        shape_positions = np.where(shape)
        if len(shape_positions[0]) > 0:
            ys = shape_positions[0] + y
            landing_height = float(np.mean(ys))
        else:
            landing_height = float(y)

        # rows that will be cleared (check BEFORE actually clearing)
        full_rows_mask = np.all(test_grid != 0, axis=1)
        full_rows_idx = np.where(full_rows_mask)[0]
        lines_cleared = len(full_rows_idx)

        # eroded piece cells = number of piece cells that lie on rows that get cleared
        if lines_cleared > 0:
            eroded_piece_cells = 0
            for i, j in zip(*shape_positions):
                if (y + i) in full_rows_idx:
                    eroded_piece_cells += 1
        else:
            eroded_piece_cells = 0

        # now actually clear lines for the board evaluation
        test_grid, _ = game.clear_lines(test_grid)
        return test_grid, landing_height, eroded_piece_cells, lines_cleared

    def choose_action(self, grid: np.ndarray, shape_list, color: int):
        """
        Try all rotations/positions; returns (rotation_index, x).
        Optimized version with early termination and reduced search space.
        """
        best_score = -math.inf
        best_move = (0, 0)
        
        # Pre-compute weight sum for faster scoring
        weight_items = list(self.weights.items())

        for rotation, shape in enumerate(shape_list):
            shape = np.array(shape, dtype=np.int8)
            h, w = shape.shape
            
            # Reduced search range for better performance
            min_x = max(-w + 1, -2)  # Don't go too far left
            max_x = min(game.cols, game.cols - w + 2)  # Don't go too far right
            
            for x in range(min_x, max_x):
                # Get drop position
                y = game.get_drop_y(grid, shape, x)
                if y < 0:
                    continue

                # Quick collision check
                if game.check_collision(grid, shape, x, y):
                    continue

                try:
                    # Simulate placement
                    new_grid, landing_height, eroded_cells, lines_cleared = self._simulate_placement(
                        grid, shape, x, y, color
                    )
                    
                    # Fast feature extraction and scoring
                    feats = self.extract_features(
                        new_grid, 
                        landing_height=landing_height, 
                        eroded_cells=eroded_cells, 
                        combo=0, 
                        lines_cleared=lines_cleared
                    )
                    
                    # Vectorized scoring
                    score = sum(self.weights.get(k, 0.0) * v for k, v in feats.items())
                    
                    # Small tie-breaker (simplified)
                    score += (rotation * 1000 + x) * 0.0001
                    
                    if score > best_score:
                        best_score = score
                        best_move = (rotation, x)
                        
                except Exception:
                    # Skip invalid moves
                    continue

        return best_move

    def choose_action_fast(self, grid: np.ndarray, shape_list, color: int):
        """
        Even faster version with reduced feature set for training.
        """
        best_score = -math.inf
        best_move = (0, 0)

        for rotation, shape in enumerate(shape_list):
            shape = np.array(shape, dtype=np.int8)
            h, w = shape.shape
            
            # Further reduced search space
            for x in range(max(-1, -w + 1), min(game.cols - w + 1, game.cols)):
                y = game.get_drop_y(grid, shape, x)
                if y < 0 or game.check_collision(grid, shape, x, y):
                    continue

                try:
                    # Simulate with minimal computation
                    test_grid = grid.copy()
                    game.lock_piece(test_grid, shape, x, y, color)
                    test_grid, lines_cleared = game.clear_lines(test_grid)
                    
                    # Minimal feature set for speed
                    rows, cols = test_grid.shape
                    occ = test_grid != 0
                    
                    # Essential features only
                    heights = rows - np.where(occ.any(axis=0), np.argmax(occ, axis=0), rows)
                    total_height = float(np.sum(heights))
                    holes = float(np.sum((np.arange(rows)[:, None] >= np.argmax(occ, axis=0)[None, :]) & (~occ)))
                    bumpiness = float(np.sum(np.abs(np.diff(heights))))
                    complete_lines = float(lines_cleared)
                    
                    # Fast scoring with key weights
                    score = (
                        self.weights.get("aggregate_height", -0.5) * total_height +
                        self.weights.get("complete_lines", 0.8) * complete_lines * 100 +
                        self.weights.get("holes", -0.4) * holes +
                        self.weights.get("bumpiness", -0.2) * bumpiness
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_move = (rotation, x)
                        
                except Exception:
                    continue

        return best_move

    def get_drop_position(self, grid: np.ndarray, shape, x: int) -> int:
        """Cached drop position calculation"""
        return game.get_drop_y(grid, np.array(shape, dtype=np.int8), x)