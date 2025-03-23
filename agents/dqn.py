import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from replay_buffers.replay_buffer import ReplayBuffer
from models.dqn import DQN as DQNNetwork
import os

class DQN:
    def __init__(self, env, lr=0.001, gamma=0.99, epsilon=0.1, buffer_size=10000, batch_size=32, name="DQN-Agent"):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_dim = env.rows * env.cols
        output_dim = env.rows * env.cols
        self.model = DQNNetwork(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.name = name
        self.checkpoint_dir = "checkpoint"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.tensor(state.flatten(), dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)
        # print*("States: ", states.shape)
        q_values = self.model(states)
        next_q_values = self.model(next_states).detach()
        target_q_values = rewards + (1 - dones) * self.gamma * torch.max(next_q_values, dim=1)[0]
        
        loss = self.criterion(q_values.gather(1, actions.unsqueeze(1)).squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self):
        path = os.path.join(self.checkpoint_dir, f"{self.name}.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self):
        path = os.path.join(self.checkpoint_dir, f"{self.name}.pth")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            print(f"Model loaded from {path}")
        else:
            print("No saved model found.")

# if __name__ == "__main__":
#     env = GomokuEnv()
#     agent = DQN(env, epsilon=0.4)
#     agent.train(episodes=100)
#     agent.play()
