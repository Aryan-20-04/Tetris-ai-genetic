import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
import pygame
import sys
import random
from typing import List,Tuple
window=None

cols,rows=10,20
block_size=30

width=cols*block_size+120
height=rows*block_size+40

bg_color=(0,0,0)
line_color=(50,50,50)
tetromino_color={
    'T': (128, 0, 128),     # Purple
    'O': (255, 255, 0),     # Yellow
    'I': (0, 255, 255),     # Cyan
    'S': (0, 255, 0),       # Green
    'Z': (255, 0, 0),       # Red
    'J': (0, 0, 255),       # Blue
    'L': (255, 165, 0)
}
tetrominoes = {
    'T': [
        [[0,1,0],
         [1,1,1]],
        [[1,0],
         [1,1],
         [1,0]],
        [[1,1,1],
         [0,1,0]],
        [[0,1],
         [1,1],
         [0,1]]
    ],
    'O': [
        [[1,1],
         [1,1]]
    ],
    'I': [
        [[1,1,1,1]],
        [[1],
         [1],
         [1],
         [1]]
    ],
    'S': [
        [[0,1,1],
         [1,1,0]],
        [[1,0],
         [1,1],
         [0,1]]
    ],
    'Z': [
        [[1,1,0],
         [0,1,1]],
        [[0,1],
         [1,1],
         [1,0]]
    ],
    'J': [
        [[1,0,0],
         [1,1,1]],
        [[1,1],
         [1,0],
         [1,0]],
        [[1,1,1],
         [0,0,1]],
        [[0,1],
         [0,1],
         [1,1]]
    ],
    'L': [
        [[0,0,1],
         [1,1,1]],
        [[1,0],
         [1,0],
         [1,1]],
        [[1,1,1],
         [1,0,0]],
        [[1,1],
         [0,1],
         [0,1]]
    ]
}

def create_grid():
    return [[0 for _ in range(cols)]for _ in range(rows)]

def drawGrid(surface,grid):
    for row in range(rows):
        for col in range(cols):
            rect=pygame.Rect(col*block_size,row*block_size+40,block_size,block_size)
            cell=grid[row][col]
            if cell:
                pygame.draw.rect(surface, cell, rect, border_radius=6)
                pygame.draw.rect(surface, (255, 255, 255), rect, 2, border_radius=6)
            else:
                pygame.draw.rect(surface, line_color, rect, 1)

def draw_ghost_piece(surface,shape,x,y,color,grid):
    while not checkCollision(grid,shape,x,y+1):
        y+=1
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell:
                ghost_rec=pygame.Rect((x+j)*block_size,(y+i)*block_size+40,block_size,block_size)
                ghost_color=(color[0]//2,color[1]//2,color[2]//2)
                pygame.draw.rect(surface,ghost_color,ghost_rec,1)
                           
def drawTetromino(surface,shape,off_x,off_y,color):
    for i,row in enumerate(shape):
        for j,cell in enumerate(row):
            if cell:
                x=off_x+j
                y=off_y+i
                if y >= 0:
                    rect = pygame.Rect(x * block_size, y * block_size, block_size, block_size)
                    pygame.draw.rect(surface, color, rect)
                    pygame.draw.rect(surface, (255, 255, 255), rect, 1)

def checkCollision(grid,shape,off_x,off_y):
    for i,row in enumerate(shape):
        for j,cell in enumerate(row):
            if cell:
                x=off_x+j
                y=off_y+i
                if x<0 or x>=cols or y>=rows:
                    return True
                if y>=0 and grid[y][x]:
                    return True
    return False

def lockTetromino(grid,shape,off_x,off_y,color):
    for i,row in enumerate(shape):
        for j,cell in enumerate(row):
            if cell:
                x=off_x+j
                y=off_y+i
                if y>=0:
                    grid[y][x]=color
                    
def clear_lines(grid):
    new_grid=[row for row in grid if any(cell==0 for cell in row)]
    lines_cleared=rows-len(new_grid)
    for _ in range(lines_cleared):
        new_grid.insert(0,[0 for _ in range(cols)])
    return new_grid,lines_cleared

def new_tetromino() -> Tuple[list[list[list[int]]], int ,tuple[int,int,int]]:
    shape_type=random.choice(list(tetrominoes.keys()))
    shape_list=tetrominoes[shape_type]
    color=tetromino_color[shape_type]
    return shape_list,0,color

def draw_next_piece(surface,next_shape,color):
    label = font.render("Next:", True, (255, 255, 255))
    surface.blit(label, (width - 100, 5))
    for i, row in enumerate(next_shape):
        for j, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(width - 100 + j * 20, 30 + i * 20, 20, 20)
                pygame.draw.rect(surface, color, rect, border_radius=4)
                pygame.draw.rect(surface, (255, 255, 255), rect, 1)

def draw_info(surface,score,paused,game_over):
    pygame.draw.rect(surface,(30,30,30),(0,0,width,40))
    score_text=font.render(f"Score: {score}",True,(255,255,255))
    surface.blit(score_text,(10,5))
    if paused:
        pause_text=font.render("PAUSED",True,(255,0,0))
        surface.blit(pause_text,(width//2-80,5))
    if game_over:
        over_text=font.render("GAME OVER - PRESS R TO RESTART",True,(255,0,0))
        surface.blit(over_text,(width//2-220,height-30))

def draw_retro_bg(surface,stars):
    surface.fill((5,5,20))
    for star in stars:
        brightness=random.randint(180,255)
        pygame.draw.circle(surface,(brightness,brightness,255),star,1)
    for x in range(0,width,block_size):
        pygame.draw.line(surface,(20,20,40),(x,40),(x,height))
    for y in range(40,height,block_size):
        pygame.draw.line(surface,(20,20,40),(0,y),(width,y))
        
def game_loop(window):
    grid=create_grid()
    score=0
    combo=0
    level=1
    lines_cleared_total=0
    fall_time=0
    fall_speed=500
    tet_x,tet_y=3,0
    tet_list, rotation, tet_color = new_tetromino()
    next_tet_list,_, next_color = new_tetromino()
    clock=pygame.time.Clock()
    paused=False
    game_over=False
    running=True
    stars=[(random.randint(0,width-1),random.randint(0,height-1))for _ in range(100)]
    while running:
        dt=clock.tick(60)
        fall_time+=dt
        current_shape=tet_list[rotation]
        
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_p and not game_over:
                    paused=not paused
                if game_over and event.key==pygame.K_r:
                    game_loop(window)
                    return
                if not paused and not game_over:
                    if event.key == pygame.K_LEFT and not checkCollision(grid, current_shape, tet_x - 1, tet_y):
                        tet_x -= 1
                    elif event.key == pygame.K_RIGHT and not checkCollision(grid, current_shape, tet_x + 1, tet_y):
                        tet_x += 1
                    elif event.key == pygame.K_DOWN and not checkCollision(grid, current_shape, tet_x, tet_y + 1):
                        tet_y += 1
                    elif event.key == pygame.K_UP:
                        next_rotation = (rotation + 1) % len(tet_list)
                        if not checkCollision(grid, tet_list[next_rotation], tet_x, tet_y):
                            rotation = next_rotation
        if not paused and not game_over and fall_time>=fall_speed:
            if not checkCollision(grid,current_shape,tet_x,tet_y+1):
                tet_y+=1
            else:
                lockTetromino(grid,current_shape,tet_x,tet_y,tet_color)
                grid,lines=clear_lines(grid)
                if lines==1:
                    score+=100
                elif lines==2:
                    score+=300
                elif lines==3:
                    score+=500
                elif lines==4:
                    score+=800
                if lines>0:
                    score+=combo*50
                    combo+=1
                    lines_cleared_total+=lines
                else:
                    combo=0
                    
                if lines_cleared_total//10>=level:
                    level+=1
                    fall_speed=max(100,fall_speed-50)
                    
                tet_x,tet_y=3,0
                tet_list=next_tet_list
                tet_color=next_color
                rotation=0
                next_tet_list, _,next_color = new_tetromino()
                if checkCollision(grid,tet_list[rotation],tet_x,tet_y):
                    game_over=True
            fall_time=0
        draw_retro_bg(window,stars)
        drawGrid(window,grid)
        drawTetromino(window,current_shape,tet_x,tet_y,tet_color)
        draw_info(window,score,paused,game_over)
        draw_ghost_piece(window,current_shape,tet_x,tet_y,tet_color,grid)
        draw_next_piece(window,next_tet_list[0],next_color)
        pygame.display.update()
    
    pygame.quit()
    sys.exit()
    
def main():
    global window,font,big_font
    pygame.init()
    window=pygame.display.set_mode((width,height))
    pygame.display.set_caption("Tetris")
    font = pygame.font.Font("PressStart2P-Regular.ttf", 16)
    big_font = pygame.font.Font("PressStart2P-Regular.ttf", 24)

    game_loop(window)
if __name__=="__main__":
    main()