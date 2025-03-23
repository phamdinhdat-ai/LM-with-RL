from envs.gomoku_env import GomokuEnv
from agents import GomokuAgent
import numpy as np 
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = GomokuEnv(rows=15, cols=15, win_length=5, ai_opponent=False)
    agent1 = GomokuAgent(env, epsilon=0.1)
    agent1.name = "Agent 1"
    agent2 = GomokuAgent(env, epsilon=0.3)
    agent2.name = "Agent 2"
    n_games = 30
    first_agent = 0
    learn_iters = 0

    n_steps = 0

    
    losses = []
    rewards = []
    total_move_taken = 0
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(n_games):
        print("Game: ", i)
        state, _ = env.reset()
        
        done1 = False
        done2 = False
        reward1_in_games = 0
        reward2_in_games = 0
        agent1_move = 0
        agent2_move = 0
        while not done1 or done2: 
            action1 = agent1.select_action(state)
            state_, reward1, done1, _, _ = env.step(action1)
            agent1.train_step(state, action1, reward1, state_, done1)
            reward1_in_games += reward1
            state = state_
            agent1_move += 1
            action2 = agent2.select_action(state)
            state_, reward2, done2, _, _ = env.step(action2)
            agent2.train_step(state, action2, reward2, state_, done2)
            reward2_in_games += reward2
            state = state_
            agent2_move += 1
            move_taken = agent1_move + agent2_move
            
            if done1:
                print("Game over: Agent 1 wins")
                break
            elif done2:
                print("Game over: Agent 2 wins")
            
            print("Total reward earned by agent 1: ", reward1_in_games)
            print("Total reward earned by agent 2: ", reward2_in_games)
            print("Total moves taken: ", move_taken)
            rewards.append([reward1_in_games, reward2_in_games])
            ax.clear()
            ax.plot(rewards, label=f'Total Reward per Episode - {agent1.name} vs {agent2.name}')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.legend([f'{agent1.name}', f'{agent2.name}'])
            plt.pause(0.01)
        move_taken = 0
        total_move_taken = 0
        
    plt.ioff()
    plt.show()
                
                
            