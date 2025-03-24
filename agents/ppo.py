import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from envs.gomoku_env import GomokuEnv

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(self, env, lr=0.001, gamma=0.99, gae_lambda=0.95, epsilon=0.2, entropy_coeff=0.01, batch_size=64, epochs=10):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_dim = env.rows * env.cols
        output_dim = env.rows * env.cols
        self.model = ActorCritic(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.memory = []
        self.checkpoint_dir = "checkpoint"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def select_action(self, state):
        state = torch.tensor(state.flatten(), dtype=torch.float, device=self.device).unsqueeze(0)
        logits, value = self.model(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
    
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        returns = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * values[t + 1] - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return advantages, returns
    
    def update(self):
        states, actions, log_probs_old, rewards, dones, values = zip(*self.memory)
        
        advantages, returns = self.compute_gae(rewards, list(values) + [0], dones)
        
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        log_probs_old = torch.tensor(log_probs_old, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float, device=self.device)
        
        for _ in range(self.epochs):
            logits, value = self.model(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(log_probs - log_probs_old)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            value_loss = F.mse_loss(value.squeeze(), returns)
            entropy_loss = dist.entropy().mean()
            
            loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.memory = []
    
    def train(self, episodes=100):
        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                self.store_transition((state.flatten(), action, log_prob.item(), reward, done, value.item()))
                state = next_state
                total_reward += reward
                
            self.update()
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        
    def save_model(self):
        path = os.path.join(self.checkpoint_dir, "PPO-Agent.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self):
        path = os.path.join(self.checkpoint_dir, "PPO-Agent.pth")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            print(f"Model loaded from {path}")
        else:
            print("No saved model found.")

# if __name__ == "__main__":
#     env = GomokuEnv()
#     agent = PPOAgent(env)
#     agent.train(episodes=100)
#     agent.save_model()
