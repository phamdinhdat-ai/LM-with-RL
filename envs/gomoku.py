import itertools
import pygame
import random
import numpy as np

class GomokuPosition:
    dirs = (
        ((0, -1), (0, 1)), 
        ((1, 0), (-1, 0)),
        ((1, 1), (-1, -1)),
        ((1, -1), (-1, 1)),
    )

    def __init__(self, rows, cols, n_to_win, players="wb", blank="."):
        self.ply = 0
        self.rows = rows
        self.cols = cols
        self.last_move = None
        self.n_to_win = n_to_win
        self.boards = [[[0] * cols for _ in range(rows)] for i in range(2)]
        self.players = players
        self.blank = blank

    def board(self, row=None, col=None):
        if row is None and col is None:
            return self.boards[self.ply&1]
        elif col is None:
            return self.boards[self.ply&1][row]

        return self.boards[self.ply&1][row][col]

    def move(self, row, col):
        if self.in_bounds(row, col) and self.is_empty(row, col):
            self.board(row)[col] = 1
            self.ply += 1
            self.last_move = row, col
            return True

        return False

    def is_empty(self, row, col):
        return not any(board[row][col] for board in self.boards)

    def in_bounds(self, y, x):
        return y >= 0 and y < self.rows and x >= 0 and x < self.cols

    def count_from_last_move(self, dy, dx):
        if not self.last_move:
            return 0

        last_board = self.boards[(self.ply-1)&1]
        y, x = self.last_move
        run = 0

        while self.in_bounds(y, x) and last_board[y][x]:
            run += 1
            x += dx
            y += dy
        return run

    def just_won(self):
        return self.ply >= self.n_to_win * 2 - 1 and any(
            (self.count_from_last_move(*x) + 
             self.count_from_last_move(*y) - 1 >= self.n_to_win)
            for x, y in self.dirs
        )
        
    def is_draw(self):
        return self.ply >= self.rows * self.cols and not self.just_won()

    def last_player(self):
        if self.ply < 1:
            raise IndexError("no moves have been made")

        return self.players[(self.ply-1)&1]

    def char_for_cell(self, row, col):
        for i, char in enumerate(self.players):
            if self.boards[i][row][col]:
                # return char

                if char == 'w':
                  return 1
                else:
                  return -1
        
        # return self.blank
        return 0

    def to_grid(self):
        return [
            [self.char_for_cell(row, col) for col in range(self.cols)]
            for row in range(self.rows)
        ]

    def __repr__(self):
        return "\n".join([" ".join(row) for row in self.to_grid()])

    def __str__(self):
        return "\n".join([" ".join(row) for row in self.to_grid()])
    

    # NEW CODE



# if __name__ == "__main__":
#     pos = GomokuPosition(rows=4, cols=4, n_to_win=3)

#     while not pos.just_won() and not pos.is_draw():self.boards
#         print(pos, "\n")

#         try:
#             if not pos.move(*map(int, input("[row col] :: ").split())):
#                 print("try again")
#         except (ValueError, IndexError):
#             print("try again")

#     print(pos, "\n")
        
#     if pos.just_won():
#         print(pos.last_player(), "won")
#     else:
#         print("draw")



class Colors:
    BLACK = 0, 0, 0
    WHITE = 255, 255, 255
    BROWN = 205, 128, 0


class Gomoku:
    def __init__(
        self,
        size=60,
        piece_size=20,
        rows=15,
        cols=15,
        n_to_win=5,
        caption="Gomoku"
    ):
        self.rows = rows
        self.cols = cols
        self.w = rows * size
        self.h = cols * size
        self.size = size
        self.piece_size = piece_size
        self.half_size = size // 2
        self.matrix = rows * cols

        self.n_to_win = n_to_win

        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.screen.fill(Colors.WHITE)
        self.player_colors = {"w": Colors.WHITE, "b": Colors.BLACK}
        self.player_names = {"w": "White", "b": "Black"}

        self.reset()
        self.board = GomokuPosition(rows, cols, n_to_win)
        self.draw_board()

    # FOR AI
    ## START
    
    def reset(self):
        self.board = GomokuPosition(self.rows, self.cols, self.n_to_win)
        self.draw_board()

    def Board(self):
        return np.asarray(self.board.to_grid())
  

    def get_state(self):
        state = self.Board()
        state = state.reshape(-1)
        return state

    def take_action(self, index):
        
        x, y = self.index_1D_to_2D(index)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # TODO AI TURN
        # if self.board.last_player() != ai:
        # move        
        # self.action(act)
        
        done = False
        # reward = -1

        self.play_step((x,y))

        print(self.board.is_empty(x,y))

        if self.board.is_empty(x,y):

            self.action((x,y))
            reward = -1

        else:
            reward = -20
                    

        if self.board.just_won():
            reward = 100
            done = True

        if self.board.is_draw():
            reward = 0
            done = True

        return reward, done

    def index_2D_to_1D(self, act):
      x, y = act
      return y + self.cols*x
      
    def index_1D_to_2D(self, index):
      y = index % self.cols
      x = int(index / self.cols)

      return x,y

    #ENDS HERE


    def row_lines(self):
        half = self.half_size

        for y in range(half, self.h - half + self.size, self.size):
            yield (half, y), (self.w - half, y)

    def col_lines(self):
        half = self.half_size

        for x in range(half, self.w - half + self.size, self.size):
            yield (x, half), (x, self.h - half)
        
    def draw_background(self):
        rect = pygame.Rect(0, 0, self.w, self.h)
        # pygame.draw.rect(self.screen, Colors.BROWN, rect)

    def draw_lines(self):
        lines = itertools.chain(self.col_lines(), self.row_lines())

        for start, end in lines:
            pygame.draw.line(
                self.screen, 
                Colors.BLACK, 
                start, 
                end, 
                width=2
            )

    def draw_board(self):
        self.draw_background()
        self.draw_lines()
        
    def draw_piece(self, row, col):
        player = self.board.last_player()
        circle_pos = (
           col * self.size + self.half_size, 
           row * self.size + self.half_size,
        )
        pygame.draw.circle(
           self.screen, 
           self.player_colors[player], 
           circle_pos, 
           self.piece_size
        )

    def show_outcome(self):
        player = self.player_names[self.board.last_player()]
        msg = "draw!" if self.board.is_draw() else f"{player} wins!"
        font_size = self.w // 10
        font = pygame.font.Font("freesansbold.ttf", font_size)
        label = font.render(msg, True, Colors.WHITE, Colors.BLACK)
        x = self.w // 2 - label.get_width() // 2
        y = self.h // 2 - label.get_height() // 2
        self.screen.blit(label, (x, y))

    def exit_on_click(self):
        while True:
            for event in pygame.event.get():
                if (event.type == pygame.QUIT or 
                        event.type == pygame.MOUSEBUTTONDOWN):
                    pygame.quit()
                    return

    def make_move(self, x, y):
        col = x // self.size
        row = y // self.size
        print(row,col)
        if self.board.move(row, col):
            self.draw_piece(row, col)
        
    def play(self):
        pygame.time.Clock().tick(10)
        self.draw_board()
        pygame.display.update()
        while not self.board.just_won() and not self.board.is_draw():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.make_move(*event.pos)
                    pygame.display.update()                        
                    print(self.board)
                    print(self.board.last_player())
                    print(self.board.just_won())

        self.show_outcome()
        pygame.display.update()
        self.exit_on_click()

    def action(self, act):
        x, y = act
        if self.board.move(x, y):
            player = self.board.last_player()
            circle_pos = (
            y * self.size + self.half_size, 
            x * self.size + self.half_size,
            )
            pygame.draw.circle(
            self.screen, 
            self.player_colors[player], 
            circle_pos, 
            self.piece_size
            )

        self.make_move(x, y)



if __name__ == "__main__":
    games = Gomoku(rows=5, cols=5, n_to_win=3)
    games.action((1,5))
    games.action((2,5))
    games.action((1,4))
    games.action((2,4))
    games.action((1,3))    
    games.action((2,3)) 
    # games.reset()
    print(games.Board())
    games.play()