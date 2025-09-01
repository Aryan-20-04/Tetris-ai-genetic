import random
import json
import time
import os
from typing import List, Dict, Tuple, Optional
import statistics
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import psutil
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import warnings
warnings.filterwarnings('ignore')

# Try to import GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

# Mock imports for missing modules - replace with actual implementations
class CoreGameAdapter:
    def __init__(self):
        self.rows = 20
        self.cols = 10
    
    def create_grid(self):
        return [[0 for _ in range(self.cols)] for _ in range(self.rows)]
    
    def new_tetromino(self):
        # Mock tetromino generation
        shapes = [
            [[[1, 1, 1, 1]]], # I piece
            [[[1, 1], [1, 1]]], # O piece
            [[[1, 1, 1], [0, 1, 0]]], # T piece
            [[[1, 1, 1], [1, 0, 0]]], # L piece
            [[[1, 1, 1], [0, 0, 1]]], # J piece
            [[[0, 1, 1], [1, 1, 0]]], # S piece
            [[[1, 1, 0], [0, 1, 1]]], # Z piece
        ]
        shape = random.choice(shapes)
        return shape, 0, random.randint(1, 7)
    
    def checkCollision(self, grid, shape, x, y):
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell and (y + i >= len(grid) or x + j < 0 or x + j >= len(grid[0]) or grid[y + i][x + j]):
                    return True
        return False
    
    def lockTetromino(self, grid, shape, x, y, color):
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    grid[y + i][x + j] = color
    
    def clear_lines(self, grid):
        lines_cleared = 0
        new_grid = []
        for row in grid:
            if 0 in row:
                new_grid.append(row[:])
            else:
                lines_cleared += 1
        
        while len(new_grid) < len(grid):
            new_grid.insert(0, [0] * len(grid[0]))
        
        return new_grid, lines_cleared

class AIAgent:
    def __init__(self, weights):
        self.weights = weights
    
    def get_features(self, grid):
        # Mock feature extraction - implement actual features
        features = {
            "aggregate_height": sum(20 - next((i for i, cell in enumerate(col) if cell), 20) for col in zip(*grid)),
            "complete_lines": 0,
            "holes": sum(sum(1 for i, cell in enumerate(col) if cell == 0 and any(col[:i])) for col in zip(*grid)),
            "bumpiness": 0,
            "height_variance": 0,
            "max_height": max(20 - next((i for i, cell in enumerate(col) if cell), 20) for col in zip(*grid)),
            "deep_wells": 0,
            "row_transitions": 0,
            "column_transitions": 0,
            "landing_height": 0,
            "eroded_piece_cells": 0,
            "well_sums": 0,
            "hole_depth": 0,
            "combo": 0,
            "line_clear_potential": 0
        }
        return features
    
    def choose_action(self, grid, shape, color):
        # Mock action selection
        return 0, random.randint(0, 9)
    
    def choose_action_fast(self, grid, shape, color):
        return self.choose_action(grid, shape, color)
    
    def get_drop_position(self, grid, shape, x):
        for y in range(len(grid) - len(shape) + 1):
            if any(grid[y + i][x + j] for i, row in enumerate(shape) for j, cell in enumerate(row) if cell and x + j < len(grid[0])):
                return max(0, y - 1)
        return len(grid) - len(shape)

RESULTS_DIR = "ga_results"

class EnhancedGPUGeneticAlgorithm:
    def __init__(self, 
                 population_size: int = 60,  # Increased for Colab
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5,  # Increased elites
                 max_pieces: int = 80,  # More pieces per game
                 max_generation: int = 200,  # More generations
                 batch_size: int = 16,  # Larger batches for GPU
                 use_mixed_precision: bool = True):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_pieces = max_pieces
        self.max_generation = max_generation
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision
        
        # Enhanced GPU initialization
        self.use_gpu = self._init_gpu_advanced()
        self.device_info = self._get_device_info()
        
        print(f"[GPU] Status: {'✅ CUDA' if self.use_gpu else '❌ CPU only'}")
        print(f"[GPU] Device: {self.device_info}")
        
        # Enhanced parameter ranges with more sophisticated features
        self.weight_ranges = {
            "aggregate_height": (-1.0, 0.0),
            "complete_lines": (0.0, 2.0),  # Increased range
            "holes": (-1.5, 0.0),  # Penalty for holes
            "bumpiness": (-1.0, 0.0),
            "height_variance": (-1.0, 0.2),
            "max_height": (-1.5, 0.0),  # Strong penalty for height
            "deep_wells": (-1.0, 0.0),
            "row_transitions": (-1.0, 0.0),
            "column_transitions": (-1.0, 0.0),
            "landing_height": (-1.0, 0.2),
            "eroded_piece_cells": (0.0, 2.0),  # Reward for clearing
            "well_sums": (-1.0, 0.0),
            "hole_depth": (-1.5, 0.0),  # Deep holes are bad
            "combo": (0.0, 1.5),  # Combo rewards
            "line_clear_potential": (0.0, 1.5),
            # Additional advanced features
            "tetris_bonus": (0.0, 2.0),  # 4-line clear bonus
            "t_spin_bonus": (0.0, 1.0),  # T-spin rewards
            "piece_diversity": (-0.5, 0.5),  # Encourage variety
            "threat_assessment": (-1.0, 0.0),  # Penalty for dangerous states
            "efficiency": (0.0, 1.0)  # Reward efficient play
        }
        
        self.weight_keys = list(self.weight_ranges.keys())
        self.generation = 0
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
        # Enhanced seed population with more sophisticated weights
        self.known_good = [
            {
                "aggregate_height": -0.51, "complete_lines": 0.76, "holes": -0.35,
                "bumpiness": -0.18, "height_variance": -0.15, "max_height": -0.20,
                "deep_wells": -0.30, "row_transitions": -0.40, "column_transitions": -0.40,
                "landing_height": -0.05, "eroded_piece_cells": 1.0, "well_sums": -0.1,
                "hole_depth": -0.5, "combo": 0.5, "line_clear_potential": 0.2,
                "tetris_bonus": 1.5, "t_spin_bonus": 0.8, "piece_diversity": 0.1,
                "threat_assessment": -0.3, "efficiency": 0.6
            },
            {
                "aggregate_height": -0.45, "complete_lines": 0.85, "holes": -0.40,
                "bumpiness": -0.22, "height_variance": -0.12, "max_height": -0.25,
                "deep_wells": -0.25, "row_transitions": -0.35, "column_transitions": -0.35,
                "landing_height": -0.03, "eroded_piece_cells": 0.9, "well_sums": -0.08,
                "hole_depth": -0.6, "combo": 0.4, "line_clear_potential": 0.3,
                "tetris_bonus": 1.2, "t_spin_bonus": 0.6, "piece_diversity": 0.05,
                "threat_assessment": -0.4, "efficiency": 0.7
            },
            {
                "aggregate_height": -0.48, "complete_lines": 0.90, "holes": -0.38,
                "bumpiness": -0.20, "height_variance": -0.10, "max_height": -0.22,
                "deep_wells": -0.28, "row_transitions": -0.38, "column_transitions": -0.38,
                "landing_height": -0.04, "eroded_piece_cells": 0.95, "well_sums": -0.09,
                "hole_depth": -0.55, "combo": 0.45, "line_clear_potential": 0.25,
                "tetris_bonus": 1.3, "t_spin_bonus": 0.7, "piece_diversity": 0.08,
                "threat_assessment": -0.35, "efficiency": 0.65
            },
            # Add more sophisticated strategies
            {
                "aggregate_height": -0.42, "complete_lines": 1.1, "holes": -0.45,
                "bumpiness": -0.25, "height_variance": -0.08, "max_height": -0.30,
                "deep_wells": -0.35, "row_transitions": -0.32, "column_transitions": -0.32,
                "landing_height": -0.02, "eroded_piece_cells": 1.2, "well_sums": -0.12,
                "hole_depth": -0.7, "combo": 0.6, "line_clear_potential": 0.4,
                "tetris_bonus": 1.8, "t_spin_bonus": 0.9, "piece_diversity": 0.15,
                "threat_assessment": -0.5, "efficiency": 0.8
            }
        ]
        
        # Thread-local storage with memory optimization
        self.thread_local = threading.local()
        
        # Performance monitoring
        self.performance_stats = {
            "gpu_batches": 0,
            "cpu_fallbacks": 0,
            "total_games": 0,
            "avg_game_time": 0
        }

    def _init_gpu_advanced(self):
        """Advanced GPU initialization with memory management"""
        if not GPU_AVAILABLE:
            return False
        
        try:
            # Test GPU with advanced features
            device = cp.cuda.Device(0)
            print(f"[GPU] Compute capability: {device.compute_capability}")
            
            # Set memory pool for efficient allocation
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=2**30)  # 1GB limit
            
            # Test mixed precision if available
            if self.use_mixed_precision:
                try:
                    test_array = cp.array([1.0, 2.0, 3.0], dtype=cp.float16)
                    _ = cp.mean(test_array)
                    print("[GPU] Mixed precision (FP16) supported")
                except:
                    self.use_mixed_precision = False
                    print("[GPU] Mixed precision not supported, using FP32")
            
            # Test batch operations
            test_batch = cp.random.random((100, 20), dtype=cp.float32)
            _ = cp.mean(test_batch, axis=1)
            
            print("[GPU] CuPy initialized with advanced features!")
            return True
            
        except Exception as e:
            print(f"[GPU] Advanced init failed: {e}")
            return False

    def _get_device_info(self):
        """Get detailed device information"""
        if self.use_gpu:
            try:
                device = cp.cuda.Device(0)
                mem_info = cp.cuda.MemoryInfo()
                return f"GPU {device.id} ({mem_info.total // 1024**2} MB)"
            except:
                return "GPU (info unavailable)"
        else:
            return f"CPU ({psutil.cpu_count()} cores, {psutil.virtual_memory().total // 1024**3} GB RAM)"

    def get_game_instance(self):
        """Get thread-local game instance with caching"""
        if not hasattr(self.thread_local, 'game'):
            self.thread_local.game = CoreGameAdapter()
            self.thread_local.ai_cache = {}  # Cache for AI decisions
        return self.thread_local.game

    def create_random_individual(self) -> Dict[str, float]:
        """Create individual with smart initialization"""
        individual = {}
        for k, (lo, hi) in self.weight_ranges.items():
            if "bonus" in k or "combo" in k or "efficiency" in k:
                # Bias towards positive values for reward parameters
                individual[k] = random.uniform(max(0, lo), hi)
            elif "hole" in k or "height" in k or "threat" in k:
                # Bias towards negative values for penalty parameters
                individual[k] = random.uniform(lo, min(0, hi))
            else:
                individual[k] = random.uniform(lo, hi)
        return individual

    def create_initial_population(self) -> List[Dict[str, float]]:
        """Enhanced population initialization"""
        population = []
        
        # Add known good seeds
        for seed in self.known_good:
            population.append(self.normalize_individual(seed.copy()))
        
        # Add strategic variations
        for seed in self.known_good:
            for strategy in ['aggressive', 'defensive', 'balanced']:
                variant = self.create_strategic_variant(seed, strategy)
                population.append(variant)
        
        # Add smart random individuals
        while len(population) < self.population_size:
            population.append(self.create_random_individual())
        
        return population[:self.population_size]

    def create_strategic_variant(self, base: Dict[str, float], strategy: str) -> Dict[str, float]:
        """Create strategic variants of base weights"""
        variant = base.copy()
        
        if strategy == 'aggressive':
            # Focus on line clearing and combos
            variant["complete_lines"] *= 1.3
            variant["tetris_bonus"] *= 1.4
            variant["combo"] *= 1.5
            variant["max_height"] *= 0.8  # Less height penalty
            
        elif strategy == 'defensive':
            # Focus on safety and hole avoidance
            variant["holes"] *= 1.5
            variant["hole_depth"] *= 1.3
            variant["max_height"] *= 1.4
            variant["threat_assessment"] *= 1.2
            
        elif strategy == 'balanced':
            # Moderate adjustments
            for key in variant:
                variant[key] *= random.uniform(0.9, 1.1)
        
        return self.normalize_individual(variant)

    def evaluate_population_gpu_optimized(self, population: List[Dict[str, float]], 
                                        games_per_eval: int = 3) -> List[float]:
        """Fully optimized GPU population evaluation"""
        if not self.use_gpu or len(population) < 4:
            return self.evaluate_population_cpu(population, games_per_eval)
        
        try:
            # Convert entire population to GPU arrays
            pop_size = len(population)
            weight_matrix = cp.zeros((pop_size, len(self.weight_keys)), 
                                   dtype=cp.float16 if self.use_mixed_precision else cp.float32)
            
            for i, individual in enumerate(population):
                for j, key in enumerate(self.weight_keys):
                    weight_matrix[i, j] = individual[key]
            
            # Batch simulation on GPU
            fitness_scores = []
            
            # Process in optimized batches
            for batch_start in range(0, pop_size, self.batch_size):
                batch_end = min(batch_start + self.batch_size, pop_size)
                batch_weights = weight_matrix[batch_start:batch_end]
                
                batch_scores = self.evaluate_batch_gpu(batch_weights, games_per_eval)
                fitness_scores.extend(batch_scores)
                
                # Memory cleanup
                if batch_start % (self.batch_size * 4) == 0:
                    cp.get_default_memory_pool().free_all_blocks()
            
            self.performance_stats["gpu_batches"] += 1
            return fitness_scores
            
        except Exception as e:
            print(f"[Warning] GPU evaluation failed: {e}")
            self.performance_stats["cpu_fallbacks"] += 1
            return self.evaluate_population_cpu(population, games_per_eval)

    def evaluate_batch_gpu(self, weight_batch, games_per_eval: int) -> List[float]:
        """GPU batch evaluation with vectorized operations"""
        batch_size = weight_batch.shape[0]
        
        # Simulate multiple games for each individual in parallel
        fitness_results = cp.zeros(batch_size, dtype=cp.float32)
        
        for game_idx in range(games_per_eval):
            # Vectorized game simulation
            game_scores = self.simulate_games_vectorized(weight_batch)
            fitness_results += game_scores
        
        # Average and add noise for diversity
        fitness_results /= games_per_eval
        noise = cp.random.uniform(-0.5, 0.5, batch_size)
        fitness_results += noise
        
        return cp.asnumpy(fitness_results).tolist()

    def simulate_games_vectorized(self, weight_batch):
        """Vectorized game simulation on GPU"""
        batch_size = weight_batch.shape[0]
        scores = cp.zeros(batch_size, dtype=cp.float32)
        
        # Simplified vectorized simulation
        # This is a mock - in practice, you'd implement actual game logic
        for i in range(batch_size):
            # Mock game simulation with weight influence
            base_score = 1000
            weight_influence = cp.sum(weight_batch[i] * cp.array([1.0] * len(self.weight_keys)))
            scores[i] = base_score + float(weight_influence) * 100
            
            # Add random variation
            scores[i] += cp.random.normal(0, 50)
        
        return scores

    def evaluate_population_cpu(self, population: List[Dict[str, float]], 
                              games_per_eval: int = 3) -> List[float]:
        """Optimized CPU evaluation with threading"""
        max_workers = min(16, len(population))  # Colab has good CPU resources
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for individual in population:
                future = executor.submit(self.evaluate_individual_cpu, individual, games_per_eval)
                futures.append(future)
            
            fitness_scores = []
            for future in as_completed(futures):
                try:
                    score = future.result(timeout=30)  # Timeout for stuck evaluations
                    fitness_scores.append(score)
                except:
                    fitness_scores.append(-1000.0)  # Penalty for failed evaluation
        
        return fitness_scores

    def evaluate_individual_cpu(self, weights: Dict[str, float], games: int = 3) -> float:
        """Enhanced CPU individual evaluation"""
        results = []
        game = self.get_game_instance()
        
        for game_idx in range(games):
            try:
                # Add game variation
                max_pieces = self.max_pieces + random.randint(-10, 10)
                result = self.run_single_game_enhanced(weights, game, max_pieces)
                results.append(result)
                
                self.performance_stats["total_games"] += 1
                
            except Exception as e:
                # Robust error handling
                results.append(self.get_default_game_result())
        
        return self.calculate_enhanced_fitness(results)

    def run_single_game_enhanced(self, weights: Dict[str, float], 
                               game: CoreGameAdapter, max_pieces: int) -> Tuple:
        """Enhanced single game simulation with more features"""
        grid = game.create_grid()
        ai = AIAgent(weights)
        
        # Game state tracking
        score = 0
        lines_cleared_total = 0
        pieces_placed = 0
        level = 1
        combo_count = 0
        max_combo = 0
        tetris_count = 0
        t_spin_count = 0
        
        # Advanced metrics
        piece_types = [0] * 7  # Track piece diversity
        threat_level = 0
        efficiency_score = 0
        
        # Pre-generate pieces for consistency
        piece_queue = [game.new_tetromino() for _ in range(max_pieces + 10)]
        
        start_time = time.time()
        
        for piece_idx in range(min(max_pieces, len(piece_queue))):
            if piece_idx >= len(piece_queue):
                break
                
            try:
                current_shape, _, color = piece_queue[piece_idx]
                
                # AI decision with caching
                rotation_choice, x_pos = ai.choose_action_fast(grid, current_shape, color)
                
                if rotation_choice >= len(current_shape):
                    rotation_choice = 0
                
                shape = current_shape[rotation_choice]
                y_pos = ai.get_drop_position(grid, shape, x_pos)
                
                # Collision handling
                if y_pos < 0 or game.checkCollision(grid, shape, x_pos, y_pos):
                    # Try alternative positions
                    alternatives = [game.cols//2-1, game.cols//2, game.cols//2+1, 0, game.cols-1]
                    placed = False
                    
                    for alt_x in alternatives:
                        alt_y = ai.get_drop_position(grid, shape, alt_x)
                        if alt_y >= 0 and not game.checkCollision(grid, shape, alt_x, alt_y):
                            x_pos, y_pos = alt_x, alt_y
                            placed = True
                            break
                    
                    if not placed:
                        break  # Game over
                
                # Place piece and update state
                game.lockTetromino(grid, shape, x_pos, y_pos, color)
                pieces_placed += 1
                piece_types[color % 7] += 1
                
                # Clear lines and update score
                grid, lines_cleared = game.clear_lines(grid)
                lines_cleared_total += lines_cleared
                
                # Enhanced scoring
                if lines_cleared > 0:
                    line_scores = [0, 40, 100, 300, 1200]
                    base_score = line_scores[min(lines_cleared, 4)] * level
                    score += base_score
                    
                    if lines_cleared == 4:
                        tetris_count += 1
                        score += int(weights.get("tetris_bonus", 0) * 500)
                    
                    combo_count += 1
                    score += combo_count * 25 * weights.get("combo", 1.0)
                    max_combo = max(max_combo, combo_count)
                    
                    efficiency_score += lines_cleared / max(1, pieces_placed * 0.1)
                else:
                    combo_count = 0
                
                # Level progression
                if lines_cleared_total >= level * 10:
                    level += 1
                
                # Threat assessment
                heights = [20 - next((i for i, cell in enumerate(col) if cell), 20) 
                          for col in zip(*grid)]
                threat_level = max(heights) + sum(1 for h in heights if h > 15)
                
            except Exception:
                break
        
        # Calculate final metrics
        game_time = time.time() - start_time
        self.performance_stats["avg_game_time"] = (
            self.performance_stats["avg_game_time"] * 0.9 + game_time * 0.1
        )
        
        # Enhanced feature calculation
        features = ai.get_features(grid)
        piece_diversity = len([p for p in piece_types if p > 0]) / 7.0
        
        return (
            score, lines_cleared_total, pieces_placed, level,
            features["holes"], features["bumpiness"], features["aggregate_height"],
            max_combo, efficiency_score, tetris_count, t_spin_count,
            piece_diversity, threat_level, features.get("line_clear_potential", 0)
        )

    def get_default_game_result(self) -> Tuple:
        """Default result for failed games"""
        return (0, 0, 1, 0, 10.0, 5.0, 10.0, 0.0, 0.0, 0, 0, 0.1, 20.0, 0.0)

    def calculate_enhanced_fitness(self, results: List[Tuple]) -> float:
        """Enhanced fitness calculation with more sophisticated scoring"""
        if not results:
            return -1000.0
        
        # Unpack and average all metrics
        metrics = list(zip(*results))
        averages = [sum(metric) / len(metric) for metric in metrics]
        
        (avg_score, avg_lines, avg_pieces, avg_level, avg_holes, avg_bumpiness, 
         avg_height, avg_combo, avg_efficiency, avg_tetris, avg_t_spin,
         avg_diversity, avg_threat, avg_lcp) = averages
        
        # Sophisticated fitness function
        fitness = (
            # Core performance
            avg_score * 0.4 +
            avg_lines * 150 +
            avg_pieces * 5.0 +
            avg_level * 80 +
            
            # Advanced bonuses
            avg_combo * 20 +
            avg_efficiency * 25 +
            avg_tetris * 100 +
            avg_t_spin * 80 +
            avg_diversity * 30 +
            avg_lcp * 15 +
            
            # Penalties
            -avg_holes * 50 -
            avg_bumpiness * 15 -
            avg_height * 4 -
            avg_threat * 8
        )
        
        # Longevity bonus
        if avg_pieces > 40:
            fitness += (avg_pieces - 40) * 2
        
        # Stability bonus for consistent performance
        if len(results) > 1:
            score_std = statistics.stdev([r[0] for r in results])
            pieces_std = statistics.stdev([r[2] for r in results])
            stability_bonus = max(0, 100 - score_std * 0.01 - pieces_std)
            fitness += stability_bonus
        
        # Add small random noise for diversity
        fitness += random.uniform(-1.0, 1.0)
        
        return fitness

    def enhanced_crossover_gpu(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Enhanced GPU-accelerated crossover with sophisticated operations"""
        if not self.use_gpu or len(population) < 8:
            return self.enhanced_crossover_cpu(population, fitness_scores)
        
        try:
            # Convert to GPU arrays
            pop_size = len(population)
            weight_matrix = cp.zeros((pop_size, len(self.weight_keys)), dtype=cp.float32)
            
            for i, individual in enumerate(population):
                for j, key in enumerate(self.weight_keys):
                    weight_matrix[i, j] = individual[key]
            
            fitness_array = cp.array(fitness_scores, dtype=cp.float32)
            
            new_population = []
            
            # Enhanced elitism with diversity
            elite_indices = cp.argsort(fitness_array)[-self.elite_size:]
            for idx in cp.asnumpy(elite_indices):
                elite_dict = {self.weight_keys[j]: float(weight_matrix[idx, j]) 
                             for j in range(len(self.weight_keys))}
                new_population.append(elite_dict)
            
            # Add diverse high-performers
            top_quarter = cp.argsort(fitness_array)[-len(population)//4:]
            diverse_elites = self.select_diverse_individuals_gpu(weight_matrix, top_quarter, 3)
            for idx in diverse_elites:
                if len(new_population) < self.elite_size + 3:
                    elite_dict = {self.weight_keys[j]: float(weight_matrix[idx, j]) 
                                 for j in range(len(self.weight_keys))}
                    new_population.append(elite_dict)
            
            # Advanced crossover operations
            while len(new_population) < self.population_size - 5:
                # Multi-parent crossover (3 parents)
                parent_indices = self.tournament_selection_gpu_batch(fitness_array, 3)
                children = self.multi_parent_crossover_gpu(weight_matrix, parent_indices)
                
                for child_weights in children:
                    if len(new_population) >= self.population_size - 5:
                        break
                    
                    child_dict = {self.weight_keys[j]: float(child_weights[j]) 
                                 for j in range(len(self.weight_keys))}
                    child_dict = self.normalize_individual(child_dict)
                    child_dict = self.enhanced_mutate(child_dict)
                    new_population.append(child_dict)
            
            # Add fresh genetic material
            while len(new_population) < self.population_size:
                new_population.append(self.create_random_individual())
            
            return new_population[:self.population_size]
            
        except Exception as e:
            print(f"[Warning] Enhanced GPU crossover failed: {e}")
            return self.enhanced_crossover_cpu(population, fitness_scores)

    def select_diverse_individuals_gpu(self, weight_matrix, candidate_indices, num_select):
        """Select diverse individuals using GPU distance calculations"""
        if len(candidate_indices) <= num_select:
            return cp.asnumpy(candidate_indices).tolist()
        
        selected = [candidate_indices[0]]  # Start with best
        
        for _ in range(num_select - 1):
            max_min_dist = -1
            best_candidate = None
            
            for candidate in candidate_indices:
                if candidate in selected:
                    continue
                
                # Calculate minimum distance to selected individuals
                candidate_weights = weight_matrix[candidate]
                min_dist = float('inf')
                
                for selected_idx in selected:
                    selected_weights = weight_matrix[selected_idx]
                    dist = cp.linalg.norm(candidate_weights - selected_weights)
                    min_dist = min(min_dist, float(dist))
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
        
        return cp.asnumpy(cp.array(selected)).tolist()

    def tournament_selection_gpu_batch(self, fitness_array, num_parents, tournament_size=5):
        """Batch tournament selection on GPU"""
        parent_indices = []
        
        for _ in range(num_parents):
            tournament_indices = cp.random.choice(len(fitness_array), 
                                                size=tournament_size, replace=False)
            tournament_fitness = fitness_array[tournament_indices]
            winner_pos = cp.argmax(tournament_fitness)
            parent_indices.append(int(tournament_indices[winner_pos]))
        
        return parent_indices

    def multi_parent_crossover_gpu(self, weight_matrix, parent_indices):
        """Advanced multi-parent crossover on GPU"""
        num_parents = len(parent_indices)
        num_weights = weight_matrix.shape[1]
        
        # Create two children using weighted combination
        children = []
        
        for child_idx in range(2):
            # Random weights for parents (favor fitter parents)
            weights = cp.random.exponential(1.0, num_parents)
            weights = weights / cp.sum(weights)
            
            # Weighted combination
            child_weights = cp.zeros(num_weights)
            for i, parent_idx in enumerate(parent_indices):
                child_weights += weights[i] * weight_matrix[parent_idx]
            
            # Add crossover variation
            if random.random() < self.crossover_rate:
                # Segment-based crossover
                segment_size = num_weights // 3
                start_idx = random.randint(0, num_weights - segment_size)
                
                # Replace segment with random parent's segment
                random_parent_idx = random.choice(parent_indices)
                child_weights[start_idx:start_idx + segment_size] = \
                    weight_matrix[random_parent_idx][start_idx:start_idx + segment_size]
            
            children.append(cp.asnumpy(child_weights))
        
        return children

    def enhanced_crossover_cpu(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Enhanced CPU crossover with sophisticated genetic operations"""
        new_population = []
        
        # Enhanced elitism with clustering
        sorted_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], reverse=True)
        
        # Add diverse elites
        elites = self.select_diverse_elites(population, sorted_indices, self.elite_size + 2)
        new_population.extend(elites)
        
        # Advanced breeding strategies
        while len(new_population) < self.population_size - 3:
            strategy = random.choice(['blend', 'uniform', 'adaptive', 'multi_parent'])
            
            if strategy == 'multi_parent':
                # 3-parent crossover
                parents = [self.tournament_selection_enhanced(population, fitness_scores) 
                          for _ in range(3)]
                children = self.multi_parent_crossover_cpu(parents)
            else:
                # 2-parent crossover
                parent1 = self.tournament_selection_enhanced(population, fitness_scores)
                parent2 = self.tournament_selection_enhanced(population, fitness_scores)
                children = self.crossover_pair_enhanced(parent1, parent2, strategy)
            
            for child in children:
                if len(new_population) >= self.population_size - 3:
                    break
                child = self.enhanced_mutate(child)
                new_population.append(child)
        
        # Add immigrants for diversity
        while len(new_population) < self.population_size:
            new_population.append(self.create_random_individual())
        
        return new_population[:self.population_size]

    def select_diverse_elites(self, population, sorted_indices, num_elites):
        """Select diverse elite individuals"""
        elites = []
        candidates = sorted_indices[:min(len(sorted_indices), num_elites * 3)]
        
        # Always include the best
        elites.append(population[candidates[0]].copy())
        
        # Select diverse remaining elites
        for _ in range(num_elites - 1):
            best_candidate = None
            max_min_distance = -1
            
            for candidate_idx in candidates:
                if any(self.individual_distance(population[candidate_idx], elite) < 0.1 
                      for elite in elites):
                    continue
                
                min_distance = min(self.individual_distance(population[candidate_idx], elite) 
                                 for elite in elites)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                elites.append(population[best_candidate].copy())
            elif len(candidates) > len(elites):
                # Fallback to next best
                elites.append(population[candidates[len(elites)]].copy())
        
        return elites

    def individual_distance(self, ind1, ind2):
        """Calculate Euclidean distance between two individuals"""
        distance = 0
        for key in self.weight_keys:
            distance += (ind1.get(key, 0) - ind2.get(key, 0)) ** 2
        return distance ** 0.5

    def tournament_selection_enhanced(self, population, fitness_scores, tournament_size=6):
        """Enhanced tournament selection with pressure adjustment"""
        # Adaptive tournament size based on generation
        adaptive_size = max(3, tournament_size - self.generation // 30)
        
        tournament_indices = random.sample(range(len(population)), 
                                         min(adaptive_size, len(population)))
        winner_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[winner_index].copy()

    def multi_parent_crossover_cpu(self, parents):
        """Multi-parent crossover for CPU"""
        num_parents = len(parents)
        children = []
        
        for child_idx in range(2):
            child = {}
            
            for key in self.weight_keys:
                # Weighted average with random weights
                weights = [random.expovariate(1.0) for _ in range(num_parents)]
                weight_sum = sum(weights)
                weights = [w / weight_sum for w in weights]
                
                child[key] = sum(weights[i] * parents[i][key] for i in range(num_parents))
            
            child = self.normalize_individual(child)
            children.append(child)
        
        return children

    def crossover_pair_enhanced(self, parent1, parent2, strategy):
        """Enhanced pairwise crossover with multiple strategies"""
        child1, child2 = {}, {}
        
        if strategy == 'blend':
            # Blend crossover with adaptive alpha
            alpha = 0.1 + 0.4 * (1.0 - self.generation / self.max_generation)
            
            for key in self.weight_keys:
                v1, v2 = parent1[key], parent2[key]
                range_val = abs(v2 - v1)
                
                low = min(v1, v2) - alpha * range_val
                high = max(v1, v2) + alpha * range_val
                
                child1[key] = random.uniform(low, high)
                child2[key] = random.uniform(low, high)
        
        elif strategy == 'uniform':
            # Uniform crossover
            for key in self.weight_keys:
                if random.random() < 0.5:
                    child1[key] = parent1[key]
                    child2[key] = parent2[key]
                else:
                    child1[key] = parent2[key]
                    child2[key] = parent1[key]
        
        elif strategy == 'adaptive':
            # Adaptive crossover based on parameter importance
            important_params = ['complete_lines', 'holes', 'aggregate_height', 'tetris_bonus']
            
            for key in self.weight_keys:
                if key in important_params:
                    # More conservative mixing for important parameters
                    alpha = random.uniform(0.3, 0.7)
                else:
                    # More aggressive mixing for less important parameters
                    alpha = random.uniform(0.1, 0.9)
                
                child1[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]
                child2[key] = alpha * parent2[key] + (1 - alpha) * parent1[key]
        
        return [self.normalize_individual(child1), self.normalize_individual(child2)]

    def enhanced_mutate(self, individual):
        """Enhanced mutation with adaptive rates and parameter-specific strategies"""
        mutated = individual.copy()
        
        # Adaptive mutation rate
        base_rate = self.mutation_rate
        generation_factor = max(0.1, 1.0 - (self.generation / (self.max_generation * 0.8)))
        adaptive_rate = base_rate * generation_factor
        
        # Parameter-specific mutation
        for key, (lo, hi) in self.weight_ranges.items():
            if random.random() < adaptive_rate:
                current_val = mutated[key]
                param_range = hi - lo
                
                # Different mutation strategies based on parameter type
                if 'bonus' in key or 'combo' in key:
                    # Conservative mutation for bonus parameters
                    variance = param_range * 0.05
                elif 'hole' in key or 'height' in key:
                    # More aggressive mutation for penalty parameters
                    variance = param_range * 0.15
                else:
                    # Standard mutation
                    variance = param_range * 0.1
                
                # Apply mutation with boundary respect
                noise = random.gauss(0, variance)
                new_val = current_val + noise
                
                # Soft boundary handling - pull back gently if out of bounds
                if new_val < lo:
                    new_val = lo + random.uniform(0, (hi - lo) * 0.1)
                elif new_val > hi:
                    new_val = hi - random.uniform(0, (hi - lo) * 0.1)
                
                mutated[key] = new_val
        
        # Occasional large mutations for exploration
        if random.random() < 0.05:  # 5% chance of large mutation
            key_to_mutate = random.choice(self.weight_keys)
            lo, hi = self.weight_ranges[key_to_mutate]
            mutated[key_to_mutate] = random.uniform(lo, hi)
        
        return mutated

    def save_progress_enhanced(self, generation, population, fitness_scores):
        """Enhanced progress saving with detailed analytics"""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        try:
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            best_individual = population[best_idx]
            best_fitness = fitness_scores[best_idx]
            
            # Save best individual with metadata
            best_data = {
                "generation": generation,
                "fitness": best_fitness,
                "weights": best_individual,
                "gpu_used": self.use_gpu,
                "performance_stats": self.performance_stats.copy(),
                "population_size": self.population_size,
                "max_pieces": self.max_pieces
            }
            
            with open(f"{RESULTS_DIR}/best_gen_{generation}.json", "w") as f:
                json.dump(best_data, f, indent=2)
            
            # Save population diversity metrics
            diversity_score = self.calculate_population_diversity(population)
            
            # Enhanced fitness history
            generation_stats = {
                "generation": generation,
                "best_fitness": best_fitness,
                "avg_fitness": statistics.mean(fitness_scores),
                "std_fitness": statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0,
                "min_fitness": min(fitness_scores),
                "diversity_score": diversity_score,
                "performance_stats": self.performance_stats.copy()
            }
            
            self.fitness_history.append(generation_stats)
            
            # Save comprehensive history
            with open(f"{RESULTS_DIR}/fitness_history.json", "w") as f:
                json.dump(self.fitness_history, f, indent=2)
            
            # Save top 5 individuals
            top_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)[:5]
            top_individuals = {
                "generation": generation,
                "top_5": [
                    {"fitness": fitness_scores[i], "weights": population[i]}
                    for i in top_indices
                ]
            }
            
            with open(f"{RESULTS_DIR}/top_individuals_gen_{generation}.json", "w") as f:
                json.dump(top_individuals, f, indent=2)
            
        except Exception as e:
            print(f"[Warning] Enhanced save failed: {e}")

    def calculate_population_diversity(self, population):
        """Calculate population diversity score"""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, min(i + 10, len(population))):  # Sample to avoid O(n²)
                distance = self.individual_distance(population[i], population[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / max(1, comparisons)

    def create_live_plot(self):
        """Create live plotting for Colab"""
        try:
            plt.style.use('dark_background')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            return fig, (ax1, ax2, ax3, ax4)
        except:
            return None, None

    def update_live_plot(self, fig, axes, generation):
        """Update live plot with current progress"""
        if fig is None or not self.fitness_history:
            return
        
        try:
            ax1, ax2, ax3, ax4 = axes
            
            # Clear previous plots
            for ax in axes:
                ax.clear()
            
            generations = [h["generation"] for h in self.fitness_history]
            best_fitness = [h["best_fitness"] for h in self.fitness_history]
            avg_fitness = [h["avg_fitness"] for h in self.fitness_history]
            diversity = [h.get("diversity_score", 0) for h in self.fitness_history]
            
            # Plot 1: Fitness Evolution
            ax1.plot(generations, best_fitness, 'g-', label='Best', linewidth=2)
            ax1.plot(generations, avg_fitness, 'b-', label='Average', alpha=0.7)
            ax1.fill_between(generations, best_fitness, avg_fitness, alpha=0.2)
            ax1.set_title('Fitness Evolution', color='white')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Diversity
            ax2.plot(generations, diversity, 'r-', linewidth=2)
            ax2.set_title('Population Diversity', color='white')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Diversity Score')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Performance Stats
            if self.performance_stats["total_games"] > 0:
                gpu_ratio = self.performance_stats["gpu_batches"] / max(1, 
                    self.performance_stats["gpu_batches"] + self.performance_stats["cpu_fallbacks"])
                
                ax3.bar(['GPU Usage', 'Games/sec'], 
                       [gpu_ratio * 100, 
                        self.performance_stats["total_games"] / max(1, self.performance_stats["avg_game_time"] * generation)])
                ax3.set_title('Performance Metrics', color='white')
                ax3.set_ylabel('Percentage / Rate')
            
            # Plot 4: Best Individual Weights
            if self.best_individual:
                weights = list(self.best_individual.values())
                keys = list(self.best_individual.keys())
                
                colors = ['red' if w < 0 else 'green' for w in weights]
                ax4.barh(range(len(weights)), weights, color=colors, alpha=0.7)
                ax4.set_yticks(range(len(keys)))
                ax4.set_yticklabels([k.replace('_', ' ').title()[:15] for k in keys], fontsize=8)
                ax4.set_title(f'Best Weights (Gen {generation})', color='white')
                ax4.set_xlabel('Weight Value')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            clear_output(wait=True)
            display(fig)
            
        except Exception as e:
            print(f"[Warning] Plot update failed: {e}")

    def run_evolution_enhanced(self, generations=200, parallel_threads=12):
        """Enhanced evolution with all optimizations"""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        print("🚀 Starting Enhanced GPU Tetris AI Evolution")
        print("="*60)
        print(f"[Config] Population: {self.population_size}")
        print(f"[Config] Max pieces per game: {self.max_pieces}")
        print(f"[Config] GPU acceleration: {self.use_gpu}")
        print(f"[Config] Mixed precision: {self.use_mixed_precision}")
        print(f"[Config] Parallel threads: {parallel_threads}")
        print(f"[Config] Device: {self.device_info}")
        print("="*60)
        
        # Setup live plotting
        fig, axes = self.create_live_plot()
        
        # Initialize population
        population = self.create_initial_population()
        
        # Evolution tracking
        stagnation_counter = 0
        best_ever = -float('inf')
        no_improvement_gens = 0
        
        # Performance monitoring
        evolution_start = time.time()
        
        try:
            for generation in range(generations):
                self.generation = generation
                gen_start = time.time()
                
                print(f"\n{'='*20} Generation {generation + 1}/{generations} {'='*20}")
                
                # Evaluate population with GPU optimization
                if self.use_gpu and len(population) >= self.batch_size:
                    fitness_scores = self.evaluate_population_gpu_optimized(population, games_per_eval=3)
                else:
                    fitness_scores = self.evaluate_population_cpu(population, games_per_eval=3)
                
                # Statistics
                max_fitness = max(fitness_scores)
                avg_fitness = statistics.mean(fitness_scores)
                std_fitness = statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0
                min_fitness = min(fitness_scores)
                
                # Track improvements
                improvement = 0
                if max_fitness > best_ever:
                    improvement = max_fitness - best_ever
                    best_ever = max_fitness
                    self.best_fitness = max_fitness
                    best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
                    self.best_individual = population[best_idx].copy()
                    stagnation_counter = 0
                    no_improvement_gens = 0
                else:
                    stagnation_counter += 1
                    no_improvement_gens += 1
                
                # Performance metrics
                gen_time = time.time() - gen_start
                total_time = time.time() - evolution_start
                
                # Print detailed progress
                print(f"🏆 Best: {max_fitness:.1f} (+{improvement:.1f}) | Avg: {avg_fitness:.1f}±{std_fitness:.1f}")
                print(f"⏱️  Gen time: {gen_time:.1f}s | Total: {total_time/60:.1f}min")
                print(f"📊 Diversity: {self.calculate_population_diversity(population):.3f}")
                print(f"🔄 Stagnation: {stagnation_counter} | GPU batches: {self.performance_stats['gpu_batches']}")
                
                # Update live plot
                if generation % 5 == 0:  # Update every 5 generations
                    self.update_live_plot(fig, axes, generation)
                
                # Save progress
                if generation % 10 == 0 or generation == generations - 1:
                    self.save_progress_enhanced(generation, population, fitness_scores)
                
                # Adaptive strategies based on progress
                if no_improvement_gens > 15:
                    print("💉 Diversity injection: Adding fresh genetic material")
                    # Replace bottom 20% with new random individuals
                    num_replace = self.population_size // 5
                    worst_indices = sorted(range(len(fitness_scores)), 
                                         key=lambda i: fitness_scores[i])[:num_replace]
                    
                    for idx in worst_indices:
                        population[idx] = self.create_random_individual()
                    
                    no_improvement_gens = 0
                
                if stagnation_counter > 25:
                    print("🔥 Population restart: Creating hybrid population")
                    # Keep top 30%, add variations, fill with random
                    num_keep = self.population_size * 3 // 10
                    top_indices = sorted(range(len(fitness_scores)), 
                                       key=lambda i: fitness_scores[i], reverse=True)[:num_keep]
                    
                    new_pop = [population[i].copy() for i in top_indices]
                    
                    # Add variations of top performers
                    while len(new_pop) < self.population_size // 2:
                        parent = random.choice(new_pop[:num_keep])
                        variant = self.enhanced_mutate(parent.copy())
                        new_pop.append(variant)
                    
                    # Fill rest with random
                    while len(new_pop) < self.population_size:
                        new_pop.append(self.create_random_individual())
                    
                    population = new_pop
                    stagnation_counter = 0
                
                # Evolution for next generation
                if generation < generations - 1:
                    if self.use_gpu:
                        population = self.enhanced_crossover_gpu(population, fitness_scores)
                    else:
                        population = self.enhanced_crossover_cpu(population, fitness_scores)
                
                # Memory cleanup
                if generation % 20 == 0:
                    gc.collect()
                    if self.use_gpu:
                        cp.get_default_memory_pool().free_all_blocks()
        
        except KeyboardInterrupt:
            print("\n🛑 Evolution interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Evolution failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Final save and summary
            total_time = time.time() - evolution_start
            
            print("\n" + "="*70)
            print("🎉 EVOLUTION COMPLETED!")
            print("="*70)
            print(f"🏅 Best fitness achieved: {self.best_fitness:.1f}")
            print(f"⏱️  Total evolution time: {total_time/60:.1f} minutes")
            print(f"🎮 Total games simulated: {self.performance_stats['total_games']}")
            print(f"⚡ GPU utilization: {self.performance_stats['gpu_batches']} batches")
            print("="*70)
            
            if self.best_individual:
                print("\n🏆 BEST WEIGHTS FOUND:")
                print("-" * 50)
                for key, value in sorted(self.best_individual.items()):
                    print(f"  {key:25s}: {value:8.4f}")
                
                # Save final best
                final_data = {
                    "final_generation": self.generation,
                    "best_fitness": self.best_fitness,
                    "best_weights": self.best_individual,
                    "evolution_time_minutes": total_time / 60,
                    "total_games": self.performance_stats["total_games"],
                    "gpu_used": self.use_gpu,
                    "performance_stats": self.performance_stats
                }
                
                with open(f"{RESULTS_DIR}/final_best.json", "w") as f:
                    json.dump(final_data, f, indent=2)
                
                print(f"\n💾 Results saved to {RESULTS_DIR}/")
        
        return self.best_individual, self.best_fitness

def setup_colab_environment():
    """Enhanced Colab environment setup"""
    print("🔧 Setting up Enhanced Colab Environment...")
    
    # Check CUDA availability
    try:
        import subprocess
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("✅ GPU detected:")
        print(nvidia_smi.stdout.split('\n')[8:12])  # GPU info lines
    except:
        print("⚠️  No GPU detected - using CPU only")
    
    # Install required packages
    packages = {
        'cupy-cuda11x': 'CuPy for GPU acceleration',
        'matplotlib': 'Plotting and visualization',
        'psutil': 'System resource monitoring',
    }
    
    for package, description in packages.items():
        try:
            __import__(package.replace('-', '_').split('-')[0])
            print(f"✅ {description}")
        except ImportError:
            print(f"📦 Installing {description}...")
            os.system(f"pip install {package}")
    
    # Create results directory with proper permissions
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Mount Google Drive for result backup
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        
        # Create backup directory in Drive
        drive_results = '/content/drive/MyDrive/tetris_ai_results'
        os.makedirs(drive_results, exist_ok=True)
        print(f"✅ Google Drive mounted - backups will be saved to {drive_results}")
        
        return drive_results
    except:
        print("ℹ️  Google Drive not available - results will be local only")
        return None

def main():
    """Enhanced main function for Colab execution"""
    # Setup environment
    drive_backup_dir = setup_colab_environment()
    
    print("\n🚀 Starting Enhanced Tetris AI Evolution in Colab")
    print("🎯 Optimized for Google Colab's GPU resources")
    
    # Create enhanced genetic algorithm
    ga = EnhancedGPUGeneticAlgorithm(
        population_size=60,      # Larger population for better diversity
        mutation_rate=0.12,      # Slightly lower for stability
        crossover_rate=0.85,     # Higher crossover rate
        elite_size=6,            # More elites preserved
        max_pieces=100,          # Longer games for better evaluation
        max_generation=300,      # More generations
        batch_size=16,           # Optimal batch size for Colab GPU
        use_mixed_precision=True # Enable FP16 for faster computation
    )
    
    # Run evolution
    best_weights, best_fitness = ga.run_evolution_enhanced(
        generations=300,
        parallel_threads=16  # Colab has good CPU resources
    )
    
    # Backup to Google Drive if available
    if drive_backup_dir:
        try:
            import shutil
            shutil.copytree(RESULTS_DIR, f"{drive_backup_dir}/latest_run", dirs_exist_ok=True)
            print(f"💾 Results backed up to Google Drive: {drive_backup_dir}")
        except Exception as e:
            print(f"⚠️  Backup failed: {e}")
    
    print("\n🎊 Evolution completed! Check the results directory for detailed analysis.")
    
    return best_weights, best_fitness

# Execute if running as main script
if __name__ == "__main__":
    best_weights, fitness = main()