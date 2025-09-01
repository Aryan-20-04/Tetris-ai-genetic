from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
import random

cols, rows = 10, 20
block_size = 30
width, height = cols * block_size, rows * block_size

def create_grid():
    return [[0 for _ in range(cols)] for _ in range(rows)]

tetrominoes = {
    'T': [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1], [1, 1], [0, 1]]],
    'O': [[[1, 1], [1, 1]]],
    'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
    'S': [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
    'Z': [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]],
    'J': [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 1, 1], [0, 0, 1]], [[0, 1], [0, 1], [1, 1]]],
    'L': [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]]]
}

tetromino_color = {
    'T': (0.5, 0, 0.5),
    'O': (1, 1, 0),
    'I': (0, 1, 1),
    'S': (0, 1, 0),
    'Z': (1, 0, 0),
    'J': (0, 0, 1),
    'L': (1, 0.65, 0)
}

class GameWidget(Widget):
    def __init__(self, score_label, **kwargs):
        super().__init__(**kwargs)
        self.score_label = score_label
        self.grid = create_grid()
        self.stars = [
            {"x": random.randint(0, width), "y": random.randint(0, height),
             "brightness": random.uniform(0.6, 1.0), "delta": random.uniform(0.005, 0.015)}
            for _ in range(100)
        ]
        self.size_hint = (None, None)
        self.size = (width, height)
        self.bind(pos=self.draw_bg, size=self.draw_bg)

        self.score = 0
        self.drop_time = 0
        self.drop_interval = 0.5
        self.rotation = 0
        self.is_game_over = False
        self.spawn_new_piece()

        Clock.schedule_interval(self.update, 1 / 60)

    def update(self, dt):
        if self.is_game_over:
            return

        self.canvas.clear()
        self.canvas.before.clear()
        self.update_stars()
        self.draw_bg()
        self.draw_grid()
        self.draw_tetromino()

        self.drop_time += dt
        if self.drop_time >= self.drop_interval:
            self.drop_time = 0
            if not self.check_collision(self.shape, self.tet_x, self.tet_y + 1):
                self.tet_y += 1
            else:
                self.lock_piece()
                self.clear_lines()
                self.spawn_new_piece()

    def update_stars(self):
        for star in self.stars:
            star["brightness"] += star["delta"]
            if star["brightness"] > 1 or star["brightness"] < 0.4:
                star["delta"] *= -1

    def draw_bg(self, *args):
        ox, oy = self.pos
        with self.canvas.before:
            Color(0.05, 0.05, 0.1)
            Rectangle(pos=self.pos, size=self.size)
            for star in self.stars:
                Color(star["brightness"], star["brightness"], 1)
                Rectangle(pos=(ox + star["x"], oy + star["y"]), size=(3, 3))
            Color(0.1, 0.1, 0.2)
            for x in range(cols + 1):
                Rectangle(pos=(ox + x * block_size, oy), size=(1, height))
            for y in range(rows + 1):
                Rectangle(pos=(ox, oy + y * block_size), size=(width, 1))

    def draw_grid(self):
        ox, oy = self.pos
        with self.canvas:
            for row in range(rows):
                for col in range(cols):
                    x = ox + col * block_size
                    y = oy + (rows - 1 - row) * block_size
                    cell = self.grid[row][col]
                    Color(*cell) if cell else Color(0.1, 0.1, 0.1)
                    Rectangle(pos=(x, y), size=(block_size, block_size))

    def draw_tetromino(self):
        ox, oy = self.pos
        with self.canvas:
            Color(*self.color)
            for i, row in enumerate(self.shape):
                for j, cell in enumerate(row):
                    if cell:
                        x = ox + (self.tet_x + j) * block_size
                        y = oy + (rows - 1 - (self.tet_y + i)) * block_size
                        Rectangle(pos=(x, y), size=(block_size, block_size))

    def move(self, direction):
        if direction == 'left' and not self.check_collision(self.shape, self.tet_x - 1, self.tet_y):
            self.tet_x -= 1
        elif direction == 'right' and not self.check_collision(self.shape, self.tet_x + 1, self.tet_y):
            self.tet_x += 1
        elif direction == 'down' and not self.check_collision(self.shape, self.tet_x, self.tet_y + 1):
            self.tet_y += 1
        elif direction == 'rotate':
            next_rotation = (self.rotation + 1) % len(tetrominoes[self.shape_type])
            next_shape = tetrominoes[self.shape_type][next_rotation]
            if not self.check_collision(next_shape, self.tet_x, self.tet_y):
                self.rotation = next_rotation
                self.shape = next_shape

    def check_collision(self, shape, x, y):
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    gx, gy = x + j, y + i
                    if gx < 0 or gx >= cols or gy >= rows or (gy >= 0 and self.grid[gy][gx]):
                        return True
        return False

    def lock_piece(self):
        for i, row in enumerate(self.shape):
            for j, cell in enumerate(row):
                if cell:
                    gx, gy = self.tet_x + j, self.tet_y + i
                    if gy >= 0:
                        self.grid[gy][gx] = self.color

    def spawn_new_piece(self):
        self.shape_type = random.choice(list(tetrominoes.keys()))
        self.rotation = 0
        self.shape = tetrominoes[self.shape_type][0]
        self.color = tetromino_color[self.shape_type]
        self.tet_x, self.tet_y = 3, 0
        if self.check_collision(self.shape, self.tet_x, self.tet_y):
            self.is_game_over = True
            self.show_game_over()

    def clear_lines(self):
        new_grid = []
        lines_cleared = 0
        for row in self.grid:
            if all(cell != 0 for cell in row):
                lines_cleared += 1
            else:
                new_grid.append(row)
        for _ in range(lines_cleared):
            new_grid.insert(0, [0] * cols)
        self.grid = new_grid
        self.score += lines_cleared * 100
        self.score_label.text = f"Score: {self.score}"

    def show_game_over(self):
        overlay = FloatLayout(size=self.size, pos=self.pos)
        overlay.name = "game_over_ui"
        with overlay.canvas:
            Color(0, 0, 0, 0.6)
            Rectangle(pos=self.pos, size=self.size)

        label = Label(text="GAME OVER", font_size=40, color=(1, 0, 0, 1),
                      size_hint=(None, None), pos_hint={"center_x": 0.5, "center_y": 0.7})
        button = Button(text="Restart", size_hint=(None, None), size=(120, 60),
                        pos_hint={"center_x": 0.5, "center_y": 0.5})
        button.bind(on_press=lambda x: self.restart_game())

        overlay.add_widget(label)
        overlay.add_widget(button)
        self.parent.add_widget(overlay)

    def restart_game(self):
        for child in self.parent.children[:]:
            if getattr(child, "name", "") == "game_over_ui":
                self.parent.remove_widget(child)

        self.grid = create_grid()
        self.score = 0
        self.score_label.text = "Score: 0"
        self.is_game_over = False
        self.spawn_new_piece()

class StarBackground(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (1, 1)
        self.stars = [
            {"x": random.randint(0, 800), "y": random.randint(0, 600),
             "brightness": random.uniform(0.6, 1.0), "delta": random.uniform(0.005, 0.02)}
            for _ in range(120)
        ]
        self.bind(size=self.on_resize, pos=self.on_resize)
        Clock.schedule_interval(self.update, 1 / 60)

    def on_resize(self, *args):
        for star in self.stars:
            star["x"] = random.randint(0, int(self.width))
            star["y"] = random.randint(0, int(self.height))

    def update(self, dt):
        self.canvas.clear()
        with self.canvas:
            Color(0.05, 0.05, 0.1)
            Rectangle(pos=self.pos, size=self.size)
            for star in self.stars:
                star["brightness"] += star["delta"]
                if star["brightness"] > 1 or star["brightness"] < 0.4:
                    star["delta"] *= -1
                Color(star["brightness"], star["brightness"], 1)
                Rectangle(pos=(self.x + star["x"], self.y + star["y"]), size=(2, 2))

class TetrisApp(App):
    def build(self):
        root = FloatLayout()

        root.add_widget(StarBackground())

        score_label = Label(text="Score: 0", size_hint=(None, None), size=(200, 50),
                            pos_hint={'x': 0.7, 'top': 0.95}, color=(1, 1, 1, 1))
        root.add_widget(score_label)

        game = GameWidget(score_label=score_label)
        game.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        root.add_widget(game)

        btn_size = (80, 80)
        controls = [
            ('L', {'x': 0.05, 'y': 0.1}, 'left'),
            ('D', {'x': 0.15, 'y': 0.1}, 'down'),
            ('R', {'x': 0.25, 'y': 0.1}, 'right'),
            ('U', {'x': 0.95, 'y': 0.1}, 'rotate'),
        ]
        for label, pos, action in controls:
            btn = Button(text=label, size_hint=(None, None), size=btn_size, pos_hint=pos)
            btn.bind(on_press=lambda x, a=action: game.move(a))
            root.add_widget(btn)

        return root

if __name__ == "__main__":
    TetrisApp().run()
