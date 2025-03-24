from envs.gomoku_env import GomokuEnv
from agents.dqn import DQN as DQNAgent
from agents.heuristic_agent import HeuristicAgent






def game_play():
    env = GomokuEnv(rows=15, cols=15, win_length=5)
    agent = DQNAgent(env, epsilon=0.4)
    agent.name = "Agent 1"
    agent.load_model()
    heuristic_agent = HeuristicAgent(env)
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            break
        action = heuristic_agent.select_action(next_state)
        next_state, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            break
    print("Game over!")
    
    
if __name__ == "__main__":
    env = GomokuEnv(rows=15, cols=15, win_length=5)
    agent = DQNAgent(env, epsilon=0.4)
    agent.name = "Agent 1"
    agent.load_model()
    env.drl_agent = agent
    env.ai_opponent = True
    env.run_pvp()
    # game_play()