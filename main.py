# from envs.gomoku_env import GomokuEnv
from agents.dqn import DQN as DQNAgent
from agents.heuristic_agent import HeuristicAgent
from agents.llm import LLMGomokuAgent
from agents.llm import train_llm_agent
import numpy as np
from envs.gomoku import GomokuEnv




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
    
    
    
# if __name__ == '__main__':
#     # --- Configuration ---
#     ROWS = 5          # Smaller board for faster training initially
#     COLS = 5
#     WIN_LENGTH = 3    # Win condition
#     NUM_EPISODES = 2000 # Increase for serious training
#     LEARNING_RATE = 2e-5 # Small learning rate typical for fine-tuning transformers
#     GAMMA = 0.98       # Discount factor
#     PRINT_INTERVAL = 20
#     RENDER_INTERVAL = 200 # Set to 0 to disable rendering

#     # Choose a smaller pre-trained model (adjust based on your hardware)
#     # 'distilbert-base-uncased' is faster, 'bert-base-uncased' is more powerful but slower
#     MODEL_NAME = 'distilbert-base-uncased'
#     # MODEL_NAME = 'bert-base-uncased' # Use if you have more resources

#     SAVE_PATH = f"./gomoku_llm_agent_{ROWS}x{COLS}_{MODEL_NAME.split('/')[-1]}"
#     LOAD_MODEL = False # Set to True to load a pre-trained agent

#     # --- Setup ---
#     env = GomokuEnv(rows=ROWS, cols=COLS, win_length=WIN_LENGTH)
#     agent = LLMGomokuAgent(MODEL_NAME, ROWS, COLS, learning_rate=LEARNING_RATE, gamma=GAMMA)

#     if LOAD_MODEL:
#         try:
#             agent.load_model(SAVE_PATH)
#         except Exception as e:
#             print(f"Could not load model from {SAVE_PATH}: {e}. Starting fresh.")

#     # --- Training ---
#     print("Starting training...")
#     rewards_history = train_llm_agent(
#         env,
#         agent,
#         num_episodes=NUM_EPISODES,
#         print_interval=PRINT_INTERVAL,
#         render_interval=RENDER_INTERVAL
#         )

#     # --- Save Final Model ---
#     agent.save_model(SAVE_PATH)
#     print("Training finished and model saved.")

#     # Optional: Plot rewards
#     try:
#         import matplotlib.pyplot as plt
#         plt.plot(rewards_history)
#         plt.title('Episode Rewards Over Time')
#         plt.xlabel('Episode')
#         plt.ylabel('Total Reward')
#         plt.savefig(f"{SAVE_PATH}_rewards.png")
#         # plt.show() # Uncomment to display the plot
#         print(f"Reward plot saved to {SAVE_PATH}_rewards.png")
#     except ImportError:
#         print("Matplotlib not found. Skipping reward plotting.")