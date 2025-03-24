import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random

class GomokuEnv(gym.Env):
    def __init__(self, rows=15, cols=15, win_length=5, ai_opponent=False, drl_agent=None):
        super(GomokuEnv, self).__init__()
        
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.ai_opponent = ai_opponent
        self.drl_agent = drl_agent
        
        self.action_space = spaces.Discrete(self.rows * self.cols)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.rows, self.cols), dtype=np.int8)
        
        self.reset()
        self.init_pygame()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        
        return self.board.copy(), {}

    def actiton_to_position(self, action):
        row = action % self.rows
        col = action // self.cols
        return row, col
        
    
    def step(self, action):
        row, col = divmod(action, self.cols)
        # print("Row: ", row)
        # print("Col: ", col)
        reward = 5
        if self.board[row, col] != 0:
            print("Invalid move")
            reward = -200
            return self.board.copy(), reward, True, False, {}
        
        self.board[row, col] = self.current_player
        done, winner = self.check_winner(row, col)
        
        if done:
            reward = 50 if winner == self.current_player else -100
            return self.board.copy(), reward, True, False, {}
        
        if np.all(self.board != 0):
            return self.board.copy(), reward, True, False, {}
        
        self.current_player *= -1
        
        if self.ai_opponent and self.current_player == -1:
            self.ai_move()
            done, winner = self.check_winner(row, col)
            if done:
                reward = 50 if winner == self.current_player else -100
                return self.board.copy(), reward, True, False, {}
            
        # print("Reward: ", reward)
        return self.board.copy(), reward, False, False, {}
    
    def ai_move(self):
        if self.drl_agent:
            state = self.board.copy()
            action = self.drl_agent.select_action(state)
            row, col = divmod(action, self.cols)
            # check if the move is valid
            if self.board[row, col] != 0:
                print("Invalid move")
                empty_positions = np.argwhere(self.board == 0)
                row, col = random.choice(empty_positions)
                
            
        else:
            empty_positions = np.argwhere(self.board == 0)
            if empty_positions.size > 0:
                row, col = random.choice(empty_positions)
        
        self.board[row, col] = self.current_player
        self.current_player *= -1
    
    def check_winner(self, row, col):
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            for d in [-1, 1]:
                r, c = row + dr * d, col + dc * d
                while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                    r += dr * d
                    c += dc * d
                    if count >= self.win_length:
                        return True, player
        
        return False, None
    
    def init_pygame(self):
        pygame.init()
        self.cell_size = 40
        self.screen = pygame.display.set_mode((self.cols * self.cell_size, self.rows * self.cell_size))
        pygame.display.set_caption("Gomoku")
        self.running = True

    def render(self):
        self.screen.fill((255, 255, 255))
        
        for r in range(self.rows):
            for c in range(self.cols):
                pygame.draw.rect(self.screen, (0, 0, 0), 
                                 (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size), 1)
                if self.board[r, c] == 1:
                    pygame.draw.circle(self.screen, (0, 0, 0), 
                                       (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2), 
                                       self.cell_size // 3)
                elif self.board[r, c] == -1:
                    pygame.draw.circle(self.screen, (255, 0, 0), 
                                       (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2), 
                                       self.cell_size // 3)
        
        pygame.display.flip()

    def run_pvp(self):
        self.reset()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    row, col = y // self.cell_size, x // self.cell_size
                    action = row * self.cols + col
                    _, _, done, _, _ = self.step(action)
                    if done:
                        running = False
            self.render()
        pygame.quit()


