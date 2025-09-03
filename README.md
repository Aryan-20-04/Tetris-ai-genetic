# Tetris AI with Genetic Algorithm

This project implements a Tetris game with AI agents that learn to play using a Genetic Algorithm (GA). It includes a headless Tetris engine for fast training, an AI agent with feature-based heuristics, and a visual demo using Pygame.

# Project Structure

main.py → Classic Tetris game with Pygame (playable manually).

core_game.py → Headless, vectorized Tetris engine for fast AI training.

adapter.py → Compatibility layer so the AI/GA can use the core_game engine.

ai_agent.py → Defines the AI agent that evaluates board states and chooses moves.

genetic_algo.py → Evolutionary training loop that optimizes the AI’s weights.

ai_play.py → Compare original vs evolved agents, and run visual AI demos.

ga_results/ → Directory where best weights and fitness history are saved.

# Features

🎲 Playable Tetris (main.py)

🧠 AI Agent with hand-crafted features:

Height, holes, bumpiness, transitions, line clears, etc.

⚡ Headless Simulation for fast training (core_game.py)

🔬 Genetic Algorithm Training

Population, crossover, mutation, elitism

Parallel evaluation with multiprocessing

Fitness based on score, survival, and efficiency

👀 Visual Demo of trained AI playing Tetris (ai_play.py)

# Installation

Make sure you have Python 3.9+ and install dependencies:

pip install pygame numpy

# Usage

Play Tetris Manually python main.py

Train the AI with Genetic Algorithm python genetic_algo.py

Saves best weights in ga_results/.

Compare Original vs Evolved Agents (Headless) python ai_play.py
Choose option 1 for headless comparison.

Run Visual Demo of Evolved AI python ai_play.py
Choose option 2 to see the AI play Tetris.

# Genetic Algorithm Parameters

Inside genetic_algo.py, you can adjust:

population_size → Number of individuals per generation

mutation_rate → Mutation probability

crossover_rate → Blend crossover probability

max_pieces → Max tetrominoes per game

max_generation → Number of generations

# Example Workflow

Run training:

python genetic_algo.py

After several generations, best weights will be saved in ga_results/best_gen_X.json.

Run:

python ai_play.py

Option 1 → Compare old vs new AI

Option 2 → Watch the AI play visually

# Notes

Training can take a while. Use fewer pieces (max_pieces) for quick testing.

Results are non-deterministic due to randomness in GA.

Best results usually appear after ~100+ generations.