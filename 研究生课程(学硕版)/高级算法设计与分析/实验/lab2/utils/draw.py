# lab2\utils\draw.py
import pygame
import math
import time
import os

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
LIGHT_GREEN = (152, 251, 152)
BLUE = (55, 173, 239)
LIGHT_BLUE = (175, 238, 238)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (250, 195, 21)
GREY = (127, 127, 127)
TURQUOISE = (64, 224, 208)
PINK = (255, 192, 203)

COLORS = {
    'closed': LIGHT_BLUE,
    'open': LIGHT_GREEN,
    'barrier': GREY,
    'desert': ORANGE,
    'river': BLUE,
    'land': WHITE,
    'start': ORANGE,
    'end': RED,
    'trace': RED,
    'back_trace': PINK
}

# LAND, RIVER, DESERT
EXTRA_COST = [0, 2, 4]


class Spot:
    def __init__(self, row, col, gap, ROWS, COLS, cost_idx=0):
        self.row = row  # 行
        self.col = col  # 列
        self.y = row * gap  # y坐标
        self.x = col * gap  # x坐标
        self.color = COLORS['land']  # 颜色
        self.cost_idx = cost_idx  # 代价
        self.neighbors = []  # 邻居
        self.gap = gap  # 间隔
        self.total_rows = ROWS  # 总行
        self.total_cols = COLS  # 总列
        self.end = False
        self.start = False
        self.is_change = True

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == COLORS['closed']

    def is_path(self):
        return self.color == PURPLE

    def is_open(self):
        return self.color == COLORS['open']

    def is_barrier(self):
        return self.color == COLORS['barrier']

    def is_start(self):
        return self.start

    def is_end(self):
        return self.end

    def reset(self):
        self.color = COLORS['land']
        self.cost_idx = 0
        self.start = False
        self.end = False
        self.is_change = True

    def is_desert(self):
        return self.cost_idx == 2

    def is_trace(self):
        return self.color == COLORS['trace']

    def is_river(self):
        return self.cost_idx == 1

    def reset_color(self):
        self.color = COLORS['land']
        if self.is_desert():
            self.color = COLORS['desert']
        elif self.is_river():
            self.color = COLORS['river']
        self.is_change = True

    def make_start(self):
        self.start = True

    def make_back_trace(self):
        self.color = COLORS['back_trace']
        self.is_change = True

    def make_end(self):
        self.end = True

    def make_closed(self):
        self.color = COLORS['closed']
        self.is_change = True

    def make_open(self):
        if not self.is_end():
            self.color = COLORS['open']
        self.is_change = True

    def make_barrier(self):
        self.color = COLORS['barrier']
        self.cost_idx = 0
        self.is_change = True

    def make_path(self, back=False):
        if not self.is_start() and not self.is_end():
            self.color = PURPLE
        if back:
            self.color = COLORS['back_trace']
        self.is_change = True

    def make_trace(self):
        self.color = COLORS['trace']
        self.is_change = True

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.gap, self.gap))
        if self.is_start():
            font = pygame.font.Font(None, self.gap)
            text = font.render("S", 1, BLACK)
            textpos = text.get_rect()
            textpos.topleft = (self.x + self.gap // 4, self.y + self.gap // 4)
            win.blit(text, textpos)
        if self.is_end():
            font = pygame.font.Font(None, self.gap)
            text = font.render("T", 1, BLACK)
            textpos = text.get_rect()
            textpos.topleft = (self.x + self.gap // 4, self.y + self.gap // 4)
            win.blit(text, textpos)
        if self.cost_idx and not self.is_start() and not self.is_end():
            font = pygame.font.Font(None, self.gap)
            text = font.render(str(EXTRA_COST[self.cost_idx]), 1, BLACK)
            textpos = text.get_rect()
            textpos.topleft = (self.x + self.gap // 4, self.y + self.gap // 4)
            win.blit(text, textpos)

    def update_neighbors(self, grid):
        self.neighbors = []
        # 统一使用8方向移动
        di = [0, 0, -1, -1, -1, 1, 1, 1]
        dj = [1, -1, -1, 0, 1, -1, 0, 1]
        for k in range(8):
            ni = self.row + di[k]
            nj = self.col + dj[k]
            if 0 <= ni < self.total_rows and 0 <= nj < self.total_cols and not grid[ni][nj].is_barrier():
                self.neighbors.append(grid[ni][nj])

    def get_move_cost(self, neighbor):
        dr = abs(self.row - neighbor.row)
        dc = abs(self.col - neighbor.col)
        
        # 判断是直线移动还是斜线移动
        if dr + dc == 1:  # 直线移动
            move_dist = 1.0
        else:  # 斜线移动 (dr + dc == 2)
            move_dist = math.sqrt(2)
            
        # 地形代价
        terrain_cost = EXTRA_COST[neighbor.cost_idx]
        
        # 总代价 = 移动距离 + 地形代价
        return move_dist + terrain_cost

    def __lt__(self, other):
        return self.row * self.total_cols + self.col


def make_grid(rows, cols, gap):
    grid = []
    for i in range(rows):
        grid.append([])
        for j in range(cols):
            spot = Spot(i, j, gap, rows, cols)
            grid[i].append(spot)
    return grid


def bye():
    pygame.quit()
    exit()


def reconstruct_path(came_from, current, draw, back=False):
    current.make_path(back)
    while current in came_from:
        current = came_from[current]
        current.make_path(back)
        draw()


def draw_grid(win, rows, cols, gap):
    width = cols * gap
    height = rows * gap
    for i in range(rows + 1):
        pygame.draw.line(win, BLACK, (0, i * gap), (width, i * gap))
    for j in range(cols + 1):
        pygame.draw.line(win, BLACK, (j * gap, 0), (j * gap, height))


def draw(win, grid, rows, cols, gap):
    need_grid_redraw = False  
    
    for row in grid:
        for spot in row:
            if spot.is_change:
                spot.draw(win)
                spot.is_change = False
                need_grid_redraw = True  
                
    
    if need_grid_redraw:
        draw_grid(win, rows, cols, gap)
        pygame.display.update()


def get_clicked_pos(pos, gap):
    x, y = pos
    row = y // gap
    col = x // gap
    return row, col


def clear_trace(grid):
    for row in grid:
        for spot in row:
            if not spot.is_start() and not spot.is_end() and not spot.is_barrier():
                spot.reset_color()


def load_map(filename, reverse=False):
    grid = []
    with open(filename, 'r') as f:
        [rows, cols, gap] = [int(s) for s in f.readline().split()]
        start = None
        end = None
        start_char = 'S'
        end_char = 'T'
        if reverse:
            start_char = 'T'
            end_char = 'S'
        for i in range(rows):
            line = f.readline().split()
            grid.append([])
            for j in range(cols):
                grid[i].append(Spot(i, j, gap, rows, cols))
                cost_idx = 0
                if line[j][0] == start_char:
                    grid[i][j].make_start()
                    start = grid[i][j]
                    cost_idx = int(line[j][1:])
                elif line[j][0] == end_char:
                    grid[i][j].make_end()
                    end = grid[i][j]
                    cost_idx = int(line[j][1:])
                elif line[j] == '-1':
                    grid[i][j].make_barrier()
                else:
                    cost_idx = int(line[j])
                grid[i][j].cost_idx = cost_idx
                if grid[i][j].cost_idx:
                    grid[i][j].reset_color()
    return grid, start, end, rows, cols, gap


def save_map(filename, grid):
    with open(filename, 'w') as f:
        rows = grid[0][0].total_rows
        cols = grid[0][0].total_cols
        gap = grid[0][0].gap
        f.write("{} {} {}\n".format(rows, cols, gap))
        for row in grid:
            for spot in row:
                if spot.is_barrier():
                    s = '-1'
                else:
                    if spot.is_start():
                        s = 'S'
                    elif spot.is_end():
                        s = 'T'
                    else:
                        s = ''
                    s += str(spot.cost_idx)
                f.write(s + ' ')
            f.write('\n')


def init_map(data_path, algo_name, reverse=False):
    ROWS = 20
    COLS = 40
    GAP = 34

    if os.path.exists(data_path):
        grid, start, end, ROWS, COLS, GAP = load_map(data_path, reverse=reverse)
    else:
        start = None
        end = None
        grid = make_grid(ROWS, COLS, GAP)

    pygame.init()
    WIDTH = COLS * GAP
    HEIGHT = ROWS * GAP
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))

    if algo_name == 'astar':
        pygame.display.set_caption("A* Path Finding Algorithm")
    else:
        pygame.display.set_caption("Bidirectional A* Path Finding Algorithm")

    return WIN, ROWS, COLS, GAP, grid, start, end


def excute(algo, win, ROWS, COLS, GAP, grid, start=None, end=None, type="h1"):
    cost_idx = 0

    while True:
        draw(win, grid, ROWS, COLS, GAP)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                bye()

            if pygame.mouse.get_pressed()[0]:  # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, GAP)
                if 0 <= row < ROWS and 0 <= col < COLS:
                    spot = grid[row][col]
                    if cost_idx:
                        if not spot.is_barrier() and not spot.is_start() and not spot.is_end():
                            spot.cost_idx = cost_idx
                            spot.reset_color()
                        continue
                    if not start and spot != end:
                        start = spot
                        start.make_start()
                    elif not end and spot != start:
                        end = spot
                        end.make_end()
                    elif spot != end and spot != start:
                        spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, GAP)
                if 0 <= row < ROWS and 0 <= col < COLS:
                    spot = grid[row][col]
                    spot.reset()
                    if spot == start:
                        start = None
                    elif spot == end:
                        end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    clear_trace(grid)
                    algo(lambda: draw(win, grid, ROWS, COLS, GAP), grid, start, end, type)
                elif event.key >= pygame.K_0 and event.key <= pygame.K_9:
                    idx = event.key - pygame.K_0
                    if idx < len(EXTRA_COST):
                        cost_idx = idx
                elif event.key == pygame.K_s:
                    filename = time.strftime("%Y%m%d-%H%M%S.txt")
                    save_map(filename, grid)
                elif event.key == pygame.K_n:
                    start = None
                    end = None
                    grid = make_grid(ROWS, COLS, GAP)
                elif event.key == pygame.K_c:
                    clear_trace(grid)
                elif event.key == pygame.K_q:
                    print("触发 q")
                    bye()

    pygame.quit()
