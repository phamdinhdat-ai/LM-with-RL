from models.a2c import ActorCritic
import os
from envs.gomoku_env import GomokuEnv
import torch
from agents.a2c import A2C
from agents.ppo import PPOAgent

def train_a2c(env, agent, episodes=100):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        move_count = 0
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        while not done and move_count < env.rows * env.cols:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = state.flatten()
            next_state = next_state.flatten()
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

if __name__ == "__main__":
    env = GomokuEnv()
    agent = PPOAgent(env)
    agent.train(episodes=100)
    agent.save_model()
