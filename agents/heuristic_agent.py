import random 
import numpy as np

class HeuristicAgent:
    def __init__(self, env):
        self.env = env
    
    def select_action(self, state):
        empty_positions = np.argwhere(state == 0)
        return random.choice(empty_positions.flatten())