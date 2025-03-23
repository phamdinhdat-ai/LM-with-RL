from envs.gomoku_env import GomokuEnv
from agents.dqn import DQN as DQNAgent
from agents.heuristic_agent import HeuristicAgent
import numpy as np 
import matplotlib.pyplot as plt

def train_agents(env, agent1, agent2, episodes=100):
    rewards = []
    total_move_taken = []
    plt.ion()
    fig, ax = plt.subplots()
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        move_count = 0
        reward1_in_episode = 0
        reward2_in_episode = 0
        while not done and move_count < env.rows * env.cols:
            if move_count % 2 == 0:
                action = agent1.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                agent1.replay_buffer.push(state, action, reward, next_state, done)
                agent1.train_step()
                reward1_in_episode += reward
            else:
                action = agent2.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                agent2.replay_buffer.push(state, action, reward, next_state, done)
                agent2.train_step()
                reward2_in_episode += reward
            
            state = next_state
            move_count += 1
        rewards.append([reward1_in_episode, reward2_in_episode])
        total_move_taken.append(move_count)
        print(f"Episode {episode + 1} completed.")
        print(f"Total moves taken: {move_count}")
        print(f"Total reward earned by agent 1: {reward1_in_episode}")
        print(f"Total reward earned by agent 2: {reward2_in_episode}")
        print("---------------------------------------------------")
        ax.clear()
        ax.plot(rewards, label=f'Total Reward per Episode - {agent1.name} vs {agent2.name}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend([f'{agent1.name}', f'{agent2.name}'])
        plt.pause(0.01)
    
        if (episode + 1) % 10 == 0:
            agent1.save_model()
            agent2.save_model()
    plt.ioff()
    plt.show()
    return rewards, total_move_taken        

   

def test_agents(env, agent, heuristic_agent, episodes=10):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        move_count = 0
        
        while not done and move_count < env.rows * env.cols:
            if move_count % 2 == 0:
                action = agent.select_action(state)
            else:
                action = heuristic_agent.select_action(state)
            
            state, _, done, _, _ = env.step(action)
            move_count += 1
        
        print(f"Test Episode {episode + 1} completed.")
        
def plot_rewards(rewards, agent1_name, agent2_name):
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(rewards, label=f'Total Reward per Episode - {agent1_name} vs {agent2_name}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend([f'{agent1_name}', f'{agent2_name}'])
    plt.pause(0.01)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    env = GomokuEnv(rows=15, cols=15, win_length=5, ai_opponent=False)
    agent1 = DQNAgent(env, epsilon=0.1)
    agent1.name = "Agent 1"
    agent2 = DQNAgent(env, epsilon=0.3)
    agent2.name = "Agent 2"
    n_games = 30
    first_agent = 0
    learn_iters = 0

    n_steps = 0

    # rewards = []
    # total_move_taken = 0
    rewards, total_move_taken = train_agents(env, agent1, agent2, episodes=30)
    plot_rewards(rewards, agent1.name, agent2.name)
    test_agents(env, agent1, agent2, episodes=10)
    



# if __name__ == "__main__":
#     env = GomokuEnv(rows=15, cols=15, win_length=5, ai_opponent=False)
#     agent1 = DQNAgent(env, epsilon=0.1)
#     agent1.name = "Agent 1"
#     agent2 = DQNAgent(env, epsilon=0.3)
#     agent2.name = "Agent 2"
#     n_games = 30
#     first_agent = 0
#     learn_iters = 0

#     n_steps = 0

    
#     losses = []
#     rewards = []
#     total_move_taken = 0
#     plt.ion()
#     fig, ax = plt.subplots()
#     for i in range(n_games):
#         print("Game: ", i)
#         state, _ = env.reset()
        
#         done1 = False
#         done2 = False
#         reward1_in_games = 0
#         reward2_in_games = 0
#         agent1_move = 0
#         agent2_move = 0
#         while not done1 or done2: 
#             action1 = agent1.select_action(state)
#             state_, reward1, done1, _, _ = env.step(action1)
#             agent1.train_step(state, action1, reward1, state_, done1)
#             reward1_in_games += reward1
#             state = state_
#             agent1_move += 1
#             action2 = agent2.select_action(state)
#             state_, reward2, done2, _, _ = env.step(action2)
#             agent2.train_step(state, action2, reward2, state_, done2)
#             reward2_in_games += reward2
#             state = state_
#             agent2_move += 1
#             move_taken = agent1_move + agent2_move
            
#             if done1:
#                 print("Game over: Agent 1 wins")
#                 break
#             elif done2:
#                 print("Game over: Agent 2 wins")
            
#             print("Total reward earned by agent 1: ", reward1_in_games)
#             print("Total reward earned by agent 2: ", reward2_in_games)
#             print("Total moves taken: ", move_taken)
#             rewards.append([reward1_in_games, reward2_in_games])
#             ax.clear()
#             ax.plot(rewards, label=f'Total Reward per Episode - {agent1.name} vs {agent2.name}')
#             ax.set_xlabel('Episode')
#             ax.set_ylabel('Total Reward')
#             ax.legend([f'{agent1.name}', f'{agent2.name}'])
#             plt.pause(0.01)
#         move_taken = 0
#         total_move_taken = 0
        
#     plt.ioff()
#     plt.show()
                
                
            