import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from envs.gomoku_env import GomokuEnv
from models.a2c import ActorCritic

class A2C:
    def __init__(self, env, lr=0.001, gamma=0.99, lambda_gae=0.95, entropy_coef=0.01):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.lambda_gae = lambda_gae  # GAE parameter
        self.entropy_coef = entropy_coef  # Entropy regularization coefficient
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        input_dim = env.rows * env.cols
        output_dim = env.rows * env.cols
        self.model = ActorCritic(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.checkpoint_dir = "checkpoint"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def select_action(self, state):
        state = torch.tensor(state.flatten(), dtype=torch.float, device=self.device).unsqueeze(0)
        logits, _ = self.model(state)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages
    
    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)
        
        logits, values = self.model(states)
        _, next_values = self.model(next_states)
        next_values = next_values.detach().squeeze()
        values = values.squeeze()
        
        advantages = self.compute_gae(rewards.tolist(), values.tolist() + [next_values[-1]], dones.tolist())
        advantages = torch.tensor(advantages, dtype=torch.float, device=self.device)
        
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        policy_loss = -(action_log_probs * advantages).mean()
        value_loss = F.mse_loss(values, rewards + self.gamma * next_values * (1 - dones))
        entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
        
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self):
        path = os.path.join(self.checkpoint_dir, "A2C.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self):
        path = os.path.join(self.checkpoint_dir, "A2C.pth")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            print(f"Model loaded from {path}")
        else:
            print("No saved model found.")

def train_a2c(env, agent, episodes=100):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        move_count = 0
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        while not done and move_count < env.rows * env.cols:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            move_count += 1
        
        agent.train_step(states, actions, rewards, next_states, dones)
        print(f"Episode {episode + 1} completed.")
        
        if (episode + 1) % 10 == 0:
            agent.save_model()

# if __name__ == "__main__":
#     env = GomokuEnv()
#     agent = A2C(env)
#     train_a2c(env, agent, episodes=100)
