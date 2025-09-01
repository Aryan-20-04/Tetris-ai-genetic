import json
import os
import pygame
import sys
import time
from typing import Dict,Optional
import numpy as np

import main as game
from ai_agent import AIAgent

def load_best_weights(generation: Optional[int] = None) -> Optional[Dict[str, float]]:
    """Load the best weights from a specific generation or the latest"""
    results_dir = "ga_results"
    filepath = os.path.join(results_dir,"best_overall.json")
    
    if not os.path.exists(filepath):
        print("No GA results found. Run the genetic algorithm first.")
        return None
    
    try:
        with open(filepath, "r") as f:
            data= json.load(f)
        print(f"Loaded BEST-EVER weights from generation {data['generation']} "
              f"(fitness: {data['fitness']:.1f})")
        return data["weights"]
    
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None

def compare_agents(original_weights: Dict[str, float], evolved_weights: Dict[str, float], 
                  games: int = 5, max_pieces: int = 1000):
    """Compare original vs evolved agents"""
    print(f"\nComparing agents over {games} games (max {max_pieces} pieces each)...")
    print("-" * 60)
    
    original_results = {"scores": [], "lines": [], "pieces": [], "level": []}
    evolved_results = {"scores": [], "lines": [], "pieces": [], "level": []}
    
    for game_num in range(games):
        print(f"\nGame {game_num + 1}/{games}")
        
        # Test original agent
        print("  Testing original agent...")
        score, lines, pieces,level = run_single_test(original_weights, max_pieces)
        original_results["scores"].append(score)
        original_results["lines"].append(lines)
        original_results["pieces"].append(pieces)
        original_results["level"].append(level)
        print(f"    Original: {score} points, {lines} lines, {pieces} pieces, {level} level")
        
        # Test evolved agent
        print("  Testing evolved agent...")
        score, lines, pieces,level = run_single_test(evolved_weights, max_pieces)
        evolved_results["scores"].append(score)
        evolved_results["lines"].append(lines)
        evolved_results["pieces"].append(pieces)
        evolved_results["level"].append(level)
        print(f"    Evolved:  {score} points, {lines} lines, {pieces} pieces, {level} level")
    
    # Calculate averages
    orig_avg_score = sum(original_results["scores"]) / games
    orig_avg_lines = sum(original_results["lines"]) / games
    orig_avg_pieces = sum(original_results["pieces"]) / games
    orig_avg_level = sum(original_results["level"]) / games
    
    evol_avg_score = sum(evolved_results["scores"]) / games
    evol_avg_lines = sum(evolved_results["lines"]) / games
    evol_avg_pieces = sum(evolved_results["pieces"]) / games
    evol_avg_level = sum(evolved_results["level"]) / games
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Original':<15} {'Evolved':<15} {'Improvement':<15}")
    print("-" * 60)
    
    score_improvement = ((evol_avg_score - orig_avg_score) / orig_avg_score * 100) if orig_avg_score > 0 else 0
    lines_improvement = ((evol_avg_lines - orig_avg_lines) / orig_avg_lines * 100) if orig_avg_lines > 0 else 0
    pieces_improvement = ((evol_avg_pieces - orig_avg_pieces) / orig_avg_pieces * 100) if orig_avg_pieces > 0 else 0
    
    print(f"{'Avg Score':<15} {orig_avg_score:<15.1f} {evol_avg_score:<15.1f} {score_improvement:+.1f}%")
    print(f"{'Avg Lines':<15} {orig_avg_lines:<15.1f} {evol_avg_lines:<15.1f} {lines_improvement:+.1f}%")
    print(f"{'Avg Pieces':<15} {orig_avg_pieces:<15.1f} {evol_avg_pieces:<15.1f} {pieces_improvement:+.1f}%")

def run_single_test(weights: Dict[str, float], max_pieces: int = 1000) -> tuple[int,int,int,int]:
    """Run a single game test without display"""
    try:
        grid = game.create_grid()
        current_shape, rotation, color = game.new_tetromino()
        next_shape, _, next_color = game.new_tetromino()
        
        ai = AIAgent(weights)
        score = 0
        lines_cleared_total = 0
        pieces_placed = 0
        level=1
        
        while pieces_placed < max_pieces:
            # Get AI move
            rotation_choice, x_pos = ai.choose_action(to_numeric_grid(grid), current_shape, hash(color)%8)
            
            # Validate rotation choice
            if rotation_choice >= len(current_shape):
                rotation_choice = 0
            
            shape = current_shape[rotation_choice]
            y_pos = ai.get_drop_position(to_numeric_grid(grid), shape, x_pos)
            
            # Check if move is valid
            if game.checkCollision(grid, shape, x_pos, y_pos):
                # Try fallback move
                x_pos = game.cols // 2
                y_pos = ai.get_drop_position(to_numeric_grid(grid), current_shape[0], x_pos)
                
                if game.checkCollision(grid, current_shape[0], x_pos, y_pos):
                    break
                
                shape = current_shape[0]
            
            # Place piece
            game.lockTetromino(grid, shape, x_pos, y_pos, color)
            pieces_placed += 1
            
            # Clear lines
            grid, lines_cleared = game.clear_lines(grid)
            lines_cleared_total += lines_cleared
            
            # Update score
            if lines_cleared == 1:
                score += 100
            elif lines_cleared == 2:
                score += 300
            elif lines_cleared == 3:
                score += 500
            elif lines_cleared == 4:
                score += 800
            if lines_cleared_total//10>=level:
                level+=1
                
            # Get next piece
            current_shape = next_shape
            color = next_color
            next_shape, _, next_color = game.new_tetromino()
            
            # Check game over
            if game.checkCollision(grid, current_shape[0], game.cols // 2, 0):
                break
        
        return score, lines_cleared_total, pieces_placed,level
    
    except Exception as e:
        print(f"Error in test: {e}")
        return 0, 0, 0,0
    
def to_numeric_grid(grid):
    """Convert grid with RGB tuples into a numeric 0/1 occupancy grid for AI"""
    return np.array([[0 if cell == 0 else 1 for cell in row] for row in grid], dtype=np.int8)


def run_visual_demo(weights: Dict[str, float]):
    """Run a visual demonstration with the evolved weights"""
    print("\nRunning visual demo with evolved weights...")
    print("Press any key to start, Ctrl+C to stop")
    
    pygame.init()
    cols, rows = 10, 20
    block_size = 30
    width = cols * block_size + 120
    height = rows * block_size + 40
    
    game.width,game.height=width,height
    game.window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Tetris AI Demo")

    # also ensure fonts exist
    if not hasattr(game, "font") or game.font is None:
        game.font = pygame.font.Font("PressStart2P-Regular.ttf", 16)
        game.big_font = pygame.font.Font("PressStart2P-Regular.ttf", 24)

    # now you can safely draw
    clock = pygame.time.Clock()

    grid = game.create_grid()
    current_shape, rotation, color = game.new_tetromino()
    next_shape, _, next_color = game.new_tetromino()
    score = 0
    lines_cleared_total = 0
    pieces_placed = 0
    game_over = False
    combo=0
    level=1
    
    ai = AIAgent(weights)
    
    start_time = time.time()
    max_speed_mode=True
    
    try:
        while not game_over:
            # Get AI decision
            rotation_choice, x_pos = ai.choose_action(to_numeric_grid(grid), current_shape, hash(color)%8)
            
            if 0 <= rotation_choice < len(current_shape):
                shape = current_shape[rotation_choice]
            else:
                shape = current_shape[0]
            
            y_pos = ai.get_drop_position(to_numeric_grid(grid), shape, x_pos)
            y_temp=0
            
            while y_temp<y_pos:
                
                if max_speed_mode:
                    y_temp=y_pos
                    game.window.fill(game.bg_color)
                    game.drawGrid(game.window,grid)
                    game.drawTetromino(game.window,shape,x_pos,y_temp,color)
                    game.draw_next_piece(game.window,next_shape[0],next_color)
                    game.draw_info(game.window,score,False,game_over)
                    
                    evolved_text=game.font.render("EVOLVED AI",True,(0,255,0))
                    lines_text=game.font.render(f"Lines: {lines_cleared_total}",True,(255,255,255))
                    pieces_text=game.font.render(f"Pieces: {pieces_placed}",True,(255,255,255))
                    game.window.blit(evolved_text,(game.width-120,70))
                    game.window.blit(lines_text,(game.width-100,100))
                    game.window.blit(pieces_text,(game.width-100,120))
                    
                    pygame.display.flip()
                    clock.tick(30)
                    y_temp+=2
                    
                else:
                    game.window.fill(game.bg_color)
                    game.drawGrid(game.window,grid)
                    game.drawTetromino(game.window,shape,x_pos,y_temp,color)
                    game.draw_next_piece(game.window,next_shape[0],next_color)
                    game.draw_info(game.window,score,False,game_over)
                    
                    evolved_text=game.font.render("EVOLVED AI",True,(0,255,0))
                    lines_text=game.font.render(f"Lines: {lines_cleared_total}",True,(255,255,255))
                    pieces_text=game.font.render(f"Pieces: {pieces_placed}",True,(255,255,255))
                    game.window.blit(evolved_text,(game.width-120,70))
                    game.window.blit(lines_text,(game.width-100,100))
                    game.window.blit(pieces_text,(game.width-100,120))
                    
                    pygame.display.flip()
                    clock.tick(5+ level*2)
                    y_temp+=1
                
                for event in pygame.event.get():
                    if event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE):
                        game_over=True
                        break
                if game_over:
                    break
            if game_over:
                break
                
            game.lockTetromino(grid, shape, x_pos, y_pos, color)
            pieces_placed += 1
                
            grid, lines_cleared = game.clear_lines(grid)
            
            if lines_cleared == 1:
                score += 100
            elif lines_cleared == 2:
                score += 300
            elif lines_cleared == 3:
                score += 500
            elif lines_cleared == 4:
                score += 800
            if lines_cleared>00:
                score+=combo*50
                combo+=1
                lines_cleared_total+=lines_cleared
            else:
                combo=0
            if lines_cleared_total // 10>=level:
                level+=1
                
            current_shape, rotation, color = next_shape, 0, next_color
            next_shape, _, next_color = game.new_tetromino()
                
            if game.checkCollision(grid, current_shape[0], game.cols // 2, 0):
                game_over = True
    except KeyboardInterrupt:
        pass
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nDemo Results:")
        print(f"Score: {score}")
        print(f"Lines: {lines_cleared_total}")
        print(f"Pieces: {pieces_placed}")
        print(f"Duration: {duration:.1f}s")
        
        pygame.quit()

def main():
    """Main function"""
    print("Tetris AI Evolution Tester")
    print("=" * 40)
    
    # Load evolved weights
    evolved_weights = load_best_weights()
    if not evolved_weights:
        return
    
    # Original weights for comparison
    original_weights = {
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
        "hole_depth": -0.5
    }
    
    while True:
        print("\nOptions:")
        print("1. Compare original vs evolved (headless)")
        print("2. Visual demo with evolved weights")
        print("3. Show evolved weights")
        print("4. Exit")
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == "1":
            compare_agents(original_weights, evolved_weights)
        elif choice == "2":
            run_visual_demo(evolved_weights)
        elif choice == "3":
            print("\nEvolved weights:")
            for key, value in evolved_weights.items():
                print(f"  {key}: {value:.6f}")
        elif choice == "4":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()