import random
import json
import time
import os
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import statistics
import numpy as np
from adapter import CoreGameAdapter
from ai_agent import AIAgent

RESULTS_DIR = "ga_results"

def evaluate_individual_worker(args):
    """Worker function for parallel evaluation"""
    weights, games, max_pieces = args
    return GeneticAlgorithm.evaluate_individual_static(weights, games, max_pieces)

class GeneticAlgorithm:
    def __init__(self, 
                 population_size: int = 30,  # Reduced from 50
                 mutation_rate: float = 0.15,  # Reduced from 0.25
                 crossover_rate: float = 0.8,
                 elite_size: int = 2,
                 max_pieces: int = 100,  # Reduced from 100
                 max_generation: int = 50):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_pieces = max_pieces
        self.max_generation = max_generation
        
        # Define the parameter ranges for each weight
        self.weight_ranges = {
            "aggregate_height": (-1.0, 0.0),
            "complete_lines": (0.0, 1.5),  # Increased upper bound
            "holes": (-1.0, 0.0),
            "bumpiness": (-1.0, 0.0),
            "height_variance": (-1.0, 0.1),
            "max_height": (-1.0, 0.0),
            "deep_wells": (-1.0, 0.0),
            "row_transitions": (-1.0, 0.0),
            "column_transitions": (-1.0, 0.0),
            "landing_height": (-1.0, 0.1),
            "eroded_piece_cells": (0.0, 1.5),  # Increased upper bound
            "well_sums": (-1.0, 0.0),
            "hole_depth": (-1.0, 0.0),
            "combo": (0.0, 1.0),
            "line_clear_potential": (0.0, 1.0)
        }
        
        self.weight_keys = list(self.weight_ranges.keys())
        self.generation = 0
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
        # Improved known good weights
        self.known_good = [
            {
                "aggregate_height": -0.51,
                "complete_lines": 0.76,
                "holes": -0.35,
                "bumpiness": -0.18,
                "height_variance": -0.15,
                "max_height": -0.20,
                "deep_wells": -0.30,
                "row_transitions": -0.40,
                "column_transitions": -0.40,
                "landing_height": -0.05,
                "eroded_piece_cells": 1.0,
                "well_sums": -0.1,
                "hole_depth": -0.5,
                "combo": 0.5,
                "line_clear_potential": 0.2
            },
            # Add a second good starting point
            {
                "aggregate_height": -0.45,
                "complete_lines": 0.85,
                "holes": -0.40,
                "bumpiness": -0.22,
                "height_variance": -0.12,
                "max_height": -0.25,
                "deep_wells": -0.25,
                "row_transitions": -0.35,
                "column_transitions": -0.35,
                "landing_height": -0.03,
                "eroded_piece_cells": 0.9,
                "well_sums": -0.08,
                "hole_depth": -0.6,
                "combo": 0.4,
                "line_clear_potential": 0.3
            }
        ]
        
        # Ensure all weights are present
        for seed in self.known_good:
            for k, (lo, hi) in self.weight_ranges.items():
                if k not in seed:
                    seed[k] = random.uniform(lo, hi)

    def create_random_individual(self) -> Dict[str, float]:
        return {k: random.uniform(lo, hi) for k, (lo, hi) in self.weight_ranges.items()}
    
    def normalize_individual(self, individual: Dict[str, float]) -> Dict[str, float]:
        return {
            k: min(max(individual.get(k, random.uniform(lo, hi)), lo), hi)
            for k, (lo, hi) in self.weight_ranges.items()
        }
    
    def create_initial_population(self) -> List[Dict[str, float]]:
        population = [self.normalize_individual(seed.copy()) for seed in self.known_good]
        
        # Add variations of good seeds
        for seed in self.known_good:
            for _ in range(3):  # 3 variations per seed
                variant = seed.copy()
                # Mutate 3-5 random parameters
                keys_to_mutate = random.sample(self.weight_keys, random.randint(3, 5))
                for k in keys_to_mutate:
                    lo, hi = self.weight_ranges[k]
                    noise = random.gauss(0, (hi - lo) * 0.1)
                    variant[k] = min(max(variant[k] + noise, lo), hi)
                population.append(variant)
        
        # Fill rest with random
        while len(population) < self.population_size:
            population.append(self.create_random_individual())
        
        return population[:self.population_size]
    
    @staticmethod
    def evaluate_individual_static(weights: Dict[str, float], games: int = 2, max_pieces: int = 50) -> float:
        """Static method for parallel processing"""
        results = []
        for _ in range(games):
            try:
                result = GeneticAlgorithm.run_single_game_static(weights, max_pieces)
                results.append(result)
            except Exception as e:
                # Return poor fitness on error
                results.append((0, 0, 1, 0, 10.0, 5.0, 10.0, 0.0, 0.0))
        
        if not results:
            return -1000.0
        
        # Calculate averages
        avg_score = sum(r[0] for r in results) / len(results)
        avg_lines = sum(r[1] for r in results) / len(results)
        avg_pieces = sum(r[2] for r in results) / len(results)
        avg_level = sum(r[3] for r in results) / len(results)
        avg_holes = sum(r[4] for r in results) / len(results)
        avg_bumpiness = sum(r[5] for r in results) / len(results)
        avg_height = sum(r[6] for r in results) / len(results)
        avg_combo = sum(r[7] for r in results) / len(results)
        avg_lcp = sum(r[8] for r in results) / len(results)
        
        # Improved fitness calculation
        fitness = (
            avg_score * 0.5 +  # Reduced weight on raw score
            avg_lines * 100 +  # Lines are very important
            avg_pieces * 3.0 +  # Survival is important
            avg_level * 50 +
            avg_combo * 10 +
            avg_lcp * 5 -
            avg_holes * 30 -  # Penalize holes
            avg_bumpiness * 10 -  # Penalize bumpiness
            avg_height * 2  # Penalize height
        )
        
        # Add small random noise to break ties
        return fitness + random.uniform(-1.0, 1.0)
    
    def evaluate_individual(self, weights: Dict[str, float], games: int = 2) -> float:
        return self.evaluate_individual_static(weights, games, self.max_pieces)
    
    @staticmethod
    def run_single_game_static(weights: Dict[str, float], max_pieces: int) -> Tuple[int, int, int, int, float, float, float, float, float]:
        """Static method for single game simulation"""
        game = CoreGameAdapter()
        grid = game.create_grid()
        ai = AIAgent(weights)
        
        # Pre-generate pieces for better performance
        piece_queue = []
        for _ in range(max_pieces + 10):  # Extra buffer
            piece_queue.append(game.new_tetromino())
        
        score = 0
        lines_cleared_total = 0
        pieces_placed = 0
        combo = 0
        level = 1
        
        # Metrics tracking
        total_combo = 0
        total_line_clear_potential = 0
        total_holes = 0
        total_bumpiness = 0
        total_height = 0
        
        current_piece_idx = 0
        current_shape, _, color = piece_queue[current_piece_idx]
        
        game_over = False
        stuck_counter = 0  # Prevent infinite loops
        
        while pieces_placed < max_pieces and not game_over:
            try:
                # Get AI decision
                rotation_choice, x_pos = ai.choose_action(grid, current_shape, color)
                
                # Validate rotation
                if rotation_choice >= len(current_shape):
                    rotation_choice = 0
                
                shape = current_shape[rotation_choice]
                y_pos = ai.get_drop_position(grid, shape, x_pos)
                
                # Check if move is valid
                if y_pos < 0 or game.checkCollision(grid, shape, x_pos, y_pos):
                    # Try center spawn position as fallback
                    x_pos = game.cols // 2 - 1
                    y_pos = ai.get_drop_position(grid, shape, x_pos)
                    if y_pos < 0 or game.checkCollision(grid, shape, x_pos, y_pos):
                        game_over = True
                        break
                
                # Place piece
                game.lockTetromino(grid, shape, x_pos, y_pos, color)
                pieces_placed += 1
                
                # Clear lines
                old_grid = grid.copy()
                grid, lines_cleared = game.clear_lines(grid)
                lines_cleared_total += lines_cleared
                
                # Scoring
                line_scores = [0, 40, 100, 300, 1200]
                if lines_cleared <= 4:
                    score += line_scores[lines_cleared] * level
                
                # Combo system
                if lines_cleared > 0:
                    combo += 1
                    score += combo * 25
                    total_combo += combo
                else:
                    combo = 0
                
                # Level progression
                if lines_cleared_total // 10 >= level:
                    level += 1
                
                # Extract features for metrics
                features = ai.get_features(grid)
                total_holes += features["holes"]
                total_bumpiness += features["bumpiness"]
                total_height += features["aggregate_height"]
                total_line_clear_potential += features.get("line_clear_potential", 0)
                
                # Get next piece
                current_piece_idx += 1
                if current_piece_idx >= len(piece_queue):
                    break
                current_shape, _, color = piece_queue[current_piece_idx]
                
                # Check if next piece can spawn
                test_shape = current_shape[0]  # Default rotation
                spawn_x = game.cols // 2 - 1
                if game.checkCollision(grid, test_shape, spawn_x, 0):
                    game_over = True
                    break
                
                stuck_counter = 0
                
            except Exception:
                stuck_counter += 1
                if stuck_counter > 5:  # Prevent infinite loops
                    break
                continue
        
        # Calculate averages
        pieces_placed = max(1, pieces_placed)
        avg_holes = total_holes / pieces_placed
        avg_bumpiness = total_bumpiness / pieces_placed
        avg_height = total_height / pieces_placed
        avg_combo = total_combo / pieces_placed
        avg_lcp = total_line_clear_potential / pieces_placed
        
        return (score, lines_cleared_total, pieces_placed, level, 
                avg_holes, avg_bumpiness, avg_height, avg_combo, avg_lcp)

    def tournament_selection(self, population: List[Dict], fitness_scores: List[float], tournament_size: int = 3) -> Dict[str, float]:
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        winner_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[winner_index].copy()
    
    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = {}, {}
        
        # Blend crossover with adaptive alpha
        for k in self.weight_keys:
            alpha = random.uniform(0.3, 0.7)  # More conservative blending
            child1[k] = alpha * parent1[k] + (1 - alpha) * parent2[k]
            child2[k] = alpha * parent2[k] + (1 - alpha) * parent1[k]
        
        return self.normalize_individual(child1), self.normalize_individual(child2)
    
    def mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        mutated = individual.copy()
        
        # Adaptive mutation rate
        current_rate = self.mutation_rate * max(0.3, 1.0 - (self.generation / 100.0))
        
        for k, (lo, hi) in self.weight_ranges.items():
            if random.random() < current_rate:
                # Gaussian mutation with adaptive variance
                variance = (hi - lo) * max(0.05, 0.2 * (1.0 - self.generation / 50.0))
                noise = random.gauss(0, variance)
                mutated[k] = min(max(mutated[k] + noise, lo), hi)
        
        return mutated
    
    def evolve_generation(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        new_population = []
        
        # Elitism - keep best individuals
        elite_indices = sorted(range(len(fitness_scores)), 
                             key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
        for i in elite_indices:
            new_population.append(population[i].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size - 2:  # Leave room for immigrants
            parent1 = self.tournament_selection(population, fitness_scores)
            parent2 = self.tournament_selection(population, fitness_scores)
            
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Immigration - add fresh genes
        num_immigrants = min(2, self.population_size - len(new_population))
        for _ in range(num_immigrants):
            new_population.append(self.create_random_individual())
        
        return new_population[:self.population_size]

    def save_progress(self, generation: int, population: List[Dict], fitness_scores: List[float]):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        try:
            best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            
            # Save best individual
            with open(f"{RESULTS_DIR}/best_gen_{generation}.json", "w") as f:
                json.dump({
                    "generation": generation,
                    "fitness": fitness_scores[best_idx],
                    "weights": population[best_idx]
                }, f, indent=2)
            
            # Save fitness history
            with open(f"{RESULTS_DIR}/fitness_history.json", "w") as f:
                json.dump(self.fitness_history, f, indent=2)
            
            # Save top 5 individuals only to save space
            sorted_indices = sorted(range(len(fitness_scores)), 
                                  key=lambda i: fitness_scores[i], reverse=True)[:5]
            top_individuals = [{"fitness": fitness_scores[i], "weights": population[i]} 
                             for i in sorted_indices]
            
            with open(f"{RESULTS_DIR}/top5_gen_{generation}.json", "w") as f:
                json.dump(top_individuals, f, indent=2)
                
        except Exception as e:
            print(f"[Warning] Could not save progress: {e}")

    def run_evolution(self, generations: int = 50, parallel_processes: Optional[int] = None):
        if parallel_processes is None:
            parallel_processes = min(6, mp.cpu_count())
        
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Resume logic
        resume_file = f"{RESULTS_DIR}/best_interrupted.json"
        if os.path.exists(resume_file):
            try:
                choice = input("[?] Found interrupted run. Resume? (y/n): ").strip().lower()
                if choice == "y":
                    with open(resume_file, "r") as f:
                        data = json.load(f)
                    self.generation = data["generation"]
                    self.best_fitness = data["fitness"]
                    self.best_individual = data["weights"]
                    print(f"[Resume] Starting from generation {self.generation}")
                else:
                    os.remove(resume_file)
            except:
                os.remove(resume_file)
        
        # Initialize population
        population = self.create_initial_population()
        print(f"[Init] Created population of {len(population)} individuals")
        print(f"[Config] Max pieces: {self.max_pieces}, Parallel processes: {parallel_processes}")

        stagnation_counter = 0
        last_best = self.best_fitness
        
        try:
            for generation in range(self.generation, generations):
                self.generation = generation
                start_time = time.time()
                print(f"\n=== Generation {generation + 1}/{generations} ===")

                # Parallel evaluation
                games_per_eval = 2  # Reduced for speed
                eval_args = [(ind, games_per_eval, self.max_pieces) for ind in population]
                
                if parallel_processes > 1:
                    with ProcessPoolExecutor(max_workers=parallel_processes) as executor:
                        fitness_scores = list(executor.map(evaluate_individual_worker, eval_args))
                else:
                    fitness_scores = [self.evaluate_individual(ind, games_per_eval) for ind in population]

                # Statistics
                avg_fitness = statistics.mean(fitness_scores)
                max_fitness = max(fitness_scores)
                min_fitness = min(fitness_scores)
                std_fitness = statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0
                
                # Track best
                best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
                if max_fitness > self.best_fitness:
                    improvement = max_fitness - self.best_fitness
                    self.best_fitness = max_fitness
                    self.best_individual = population[best_idx].copy()
                    stagnation_counter = 0
                    print(f"*** NEW BEST! Improvement: +{improvement:.1f} ***")
                else:
                    stagnation_counter += 1

                # Record history
                self.fitness_history.append({
                    "generation": generation,
                    "avg_fitness": avg_fitness,
                    "max_fitness": max_fitness,
                    "min_fitness": min_fitness,
                    "std_fitness": std_fitness
                })

                # Print results
                elapsed = time.time() - start_time
                print(f"Time: {elapsed:.1f}s | Max: {max_fitness:.1f} | Avg: {avg_fitness:.1f} ± {std_fitness:.1f}")
                print(f"Best ever: {self.best_fitness:.1f} | Stagnation: {stagnation_counter}")
                
                # Save progress
                if generation % 5 == 0 or generation == generations - 1:  # Save every 5 generations
                    self.save_progress(generation, population, fitness_scores)

                # Anti-stagnation measures
                if stagnation_counter > 15:
                    inject_count = max(3, self.population_size // 8)
                    print(f"[Diversity Boost] Injecting {inject_count} random individuals")
                    for i in range(inject_count):
                        population[-(i+1)] = self.create_random_individual()
                    stagnation_counter = 0

                # Evolution for next generation
                if generation < generations - 1:
                    population = self.evolve_generation(population, fitness_scores)

        except KeyboardInterrupt:
            print("\n[!] Training interrupted by user")
            if self.best_individual:
                with open(f"{RESULTS_DIR}/best_interrupted.json", "w") as f:
                    json.dump({
                        "generation": self.generation,
                        "fitness": self.best_fitness,
                        "weights": self.best_individual
                    }, f, indent=2)
                print(f"[Save] Best weights saved at generation {self.generation}")

        print("\n" + "="*50)
        print("EVOLUTION COMPLETED!")
        print(f"Best fitness achieved: {self.best_fitness:.1f}")
        print("="*50)
        
        return self.best_individual, self.best_fitness


def main():
    print("Starting Tetris AI Evolution...")
    
    ga = GeneticAlgorithm(
        population_size=30,  # Reduced for faster iteration
        mutation_rate=0.15,
        crossover_rate=0.8,
        elite_size=2,
        max_pieces=100  # Reduced for faster games
    )
    
    best_weights, best_fitness = ga.run_evolution(
        generations=250, 
        parallel_processes=6
    )
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    if best_weights:
        print("Best weights found:")
        for key, value in sorted(best_weights.items()):
            print(f"  {key:20s}: {value:8.4f}")
        print(f"\nFinal fitness: {best_fitness:.2f}")
    else:
        print("No solution found!")

if __name__ == "__main__":
    main()