import random 
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = state.flatten()
        next_state = next_state.flatten()
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)
# class ReplayBuffer:
    
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = deque(maxlen=self.capacity) # tao khong gian luu tru du lieu
    
#     def push(self, state, action, reward, next_state, done): # them du lieu vao buffer
#         experience = (state, action, np.array([reward]), next_state, done)
#         self.buffer.append(experience)
    
#     def sample(self, batch_size, sequential=False):
        
#         if batch_size > len(self.buffer):
#             batch_size = len(self.buffer)
            
#         if sequential: # lay du lieu tu buffer theo thu tu
#             idx = np.random.choice(len(self.buffer) - batch_size)
#             return [self.buffer[i] for i in range(idx, idx + batch_size)]
#         else: # lay du lieu tu buffer ngau nhien
#             idx = np.random.choice(len(self.buffer), batch_size, replace=False)
#             return [self.buffer[i] for i in idx]
        
#     def clear(self):
#         self.buffer.clear()
    
#     def __len__(self):
#         return len(self.buffer)
    
    
    

