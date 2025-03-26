import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

# --- Gomoku Environment (Provided in the prompt - slightly modified for compatibility) ---
class GomokuEnv(gym.Env):
    # (Keep the GomokuEnv class exactly as provided in the previous prompt)
    # ... (rest of the GomokuEnv class code from the previous prompt) ...
    # Important: Ensure the __init__ accepts rows, cols, win_length
    def __init__(self, rows=15, cols=15, win_length=5, ai_opponent=False, drl_agent=None):
        super(GomokuEnv, self).__init__()

        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        # Note: ai_opponent and drl_agent args might not be directly used
        # if the LLM agent plays against itself or a fixed opponent.
        # Modify logic as needed for the specific training setup.

        self.action_space = spaces.Discrete(self.rows * self.cols)
        # Observation space is the board, but the LLM agent will process a string version
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.rows, self.cols), dtype=np.int8)

        # self.reset() # Reset is called externally before training usually
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.init_pygame() # Optional: Initialize pygame if rendering

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        # Return the initial board state and an empty info dict
        return self.board.copy(), {}

    def actiton_to_position(self, action):
        # Corrected logic: row is action // cols, col is action % cols
        row = action // self.cols
        col = action % self.cols
        return row, col

    def step(self, action):
        row, col = self.actiton_to_position(action)

        # Basic reward, can be refined
        reward = 0.1 # Small reward for surviving a step

        # Check for invalid move
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols or self.board[row, col] != 0:
            # print(f"Invalid move attempted: Action {action} -> ({row}, {col}) by player {self.current_player}")
            # print("Current board:\n", self.board)
            reward = -1.0 # Penalize invalid moves heavily
            # For REINFORCE, it might be better to end the episode on invalid move
            # Or just give a large penalty and let the game continue if possible (though state doesn't change)
            # Let's return done=True for invalid moves to simplify training loop
            return self.board.copy(), reward, True, False, {"error": "Invalid move"}


        # Place the piece
        self.board[row, col] = self.current_player

        # Check for winner
        done, winner = self.check_winner(row, col)
        if done:
            if winner == self.current_player:
                reward = 1.0 # Win reward
            # else: # Should not happen if check_winner is correct
            #     reward = -1.0 # Lose reward (handled when opponent wins)
            # print(f"Player {self.current_player} wins!")
            return self.board.copy(), reward, True, False, {} # Terminated = True

        # Check for draw
        if np.all(self.board != 0):
            reward = 0.5 # Draw reward (better than losing)
            # print("Draw!")
            return self.board.copy(), reward, True, False, {} # Terminated = True

        # Switch player
        self.current_player *= -1

        # Opponent's turn (if applicable, e.g., playing against random or fixed AI)
        # If training via self-play, the *next* call to step will be the opponent
        # Let's assume for now the training loop handles alternating players

        # Game continues
        # Truncated = False (no time limits here)
        return self.board.copy(), reward, False, False, {}

    def check_winner(self, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
             return False, None # Should not happen if placement is valid

        player = self.board[row, col]
        if player == 0: # Cannot win from an empty cell check
            return False, None

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)] # Horizontal, Vertical, Diag down-right, Diag down-left

        for dr, dc in directions:
            count = 1
            # Check in positive direction
            for i in range(1, self.win_length):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                else:
                    break
            # Check in negative direction
            for i in range(1, self.win_length):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
                    count += 1
                else:
                    break

            if count >= self.win_length:
                return True, player

        return False, None # No winner found

    # --- Pygame methods (optional, keep if needed) ---
    def init_pygame(self):
        try:
            pygame.init()
            self.cell_size = 30 # Smaller for potentially larger boards
            screen_width = self.cols * self.cell_size
            screen_height = self.rows * self.cell_size
            # Check if display mode is possible, otherwise disable rendering
            try:
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("Gomoku (LLM Agent Training)")
                self.pygame_initialized = True
            except pygame.error:
                 print("Pygame display could not be initialized (maybe running headless). Rendering disabled.")
                 self.pygame_initialized = False
                 self.screen = None
        except Exception as e:
             print(f"Pygame initialization failed: {e}. Rendering disabled.")
             self.pygame_initialized = False
             self.screen = None


    def render(self):
        if not self.pygame_initialized or self.screen is None:
            # print("Pygame not initialized or screen not available, cannot render.")
            return

        if not pygame.display.get_init(): # Check if display is still usable
             print("Pygame display lost. Cannot render.")
             self.pygame_initialized = False
             return

        try:
            self.screen.fill((200, 200, 200)) # Light gray background

            for r in range(self.rows):
                for c in range(self.cols):
                    # Draw grid lines
                    pygame.draw.rect(self.screen, (0, 0, 0),
                                     (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size), 1)
                    # Draw pieces
                    if self.board[r, c] == 1: # Player 1 (Black)
                        pygame.draw.circle(self.screen, (0, 0, 0),
                                           (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2),
                                           self.cell_size // 2 - 3)
                    elif self.board[r, c] == -1: # Player -1 (White)
                        pygame.draw.circle(self.screen, (255, 255, 255),
                                           (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2),
                                           self.cell_size // 2 - 3)
                        pygame.draw.circle(self.screen, (0, 0, 0), # Outline for white pieces
                                           (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2),
                                           self.cell_size // 2 - 3, 1)

            pygame.display.flip()

            # Handle window close event during rendering
            for event in pygame.event.get():
                 if event.type == pygame.QUIT:
                     print("Pygame window closed.")
                     pygame.quit()
                     self.pygame_initialized = False # Stop future rendering attempts


        except pygame.error as e:
            print(f"Pygame rendering error: {e}. Disabling rendering.")
            self.pygame_initialized = False
            pygame.quit()

    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False