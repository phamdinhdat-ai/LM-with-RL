import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
from envs.gomoku_env import GomokuEnv

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(self, env, lr=0.001, gamma=0.99, epsilon=0.1, buffer_size=10000, batch_size=64):
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
        self.name = "DQN-Agent"
        
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
        
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states).reshape(self.batch_size, -1), dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        next_states = torch.tensor(np.array(next_states).reshape(self.batch_size, -1), dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)
        
        q_values = self.model(states)
        next_q_values = self.model(next_states).detach()
        target_q_values = rewards + (1 - dones) * self.gamma * torch.max(next_q_values, dim=1)[0]
        
        loss = self.criterion(q_values.gather(1, actions.unsqueeze(1)).squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, episodes=100):
        rewards = []
        plt.ion()
        fig, ax = plt.subplots()
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            move_count = 0
            
            while not done and move_count < self.env.rows * self.env.cols:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                self.train_step()
                state = next_state
                total_reward += reward
                move_count += 1
            
            rewards.append(total_reward)
            ax.clear()
            ax.plot(rewards, label=f'Total Reward per Episode - {self.name}')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.legend()
            plt.pause(0.01)
            
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        
        plt.ioff()
        plt.show()
    
    def play(self):
        state, _ = self.env.reset()
        done = False
        while not done:
            action = self.select_action(state)
            state, _, done, _, _ = self.env.step(action)
            self.env.render()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# if __name__ == "__main__":
#     env = GomokuEnv()
#     agent = DQN(env, epsilon=0.4)
#     agent.train(episodes=100)
#     agent.play()
