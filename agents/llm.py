import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

# --- LLM Agent ---
class LLMGomokuAgent:
    def __init__(self, model_name, rows, cols, learning_rate=5e-6, gamma=0.99):
        self.rows = rows
        self.cols = cols
        self.num_actions = rows * cols
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Configure the model for classification with num_actions labels
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_actions
        ).to(self.device)

        # Optimizer (AdamW is recommended for transformers)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # REINFORCE buffers
        self.log_probs = []
        self.rewards = []

    def _board_to_string(self, board, current_player):
        """Converts the board numpy array to a string representation."""
        map_chars = {0: '.', 1: 'X', -1: 'O'}
        board_str = ""
        for r in range(self.rows):
            board_str += "".join([map_chars[board[r, c]] for c in range(self.cols)]) + "\n"
        # Add context about whose turn it is
        board_str += f"Player {'X' if current_player == 1 else 'O'}'s turn."
        return board_str

    def select_action(self, board, current_player):
        """Selects an action based on the current board state."""
        # 1. Convert board to string and tokenize
        board_str = self._board_to_string(board, current_player)
        inputs = self.tokenizer(board_str, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device) # Adjust max_length if needed

        # 2. Get logits from the model
        self.model.eval() # Set model to evaluation mode for inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0) # Remove batch dimension

        # 3. Apply action mask (prevent selecting occupied cells)
        mask = torch.from_numpy(board.flatten() == 0).float().to(self.device)
        # Ensure mask has the same size as logits
        if mask.shape[0] != logits.shape[0]:
             raise ValueError(f"Mask shape {mask.shape} doesn't match logits shape {logits.shape}")

        # Set logits of invalid actions to a very small number (-inf)
        masked_logits = logits.masked_fill(mask == 0, -float('inf'))

        # 4. Convert logits to probabilities and sample action
        if torch.isinf(masked_logits).all():
            # If all actions are masked (e.g., board full, but game didn't end?)
            # Or if initial logits were all -inf for valid moves (unlikely)
            print("Warning: All valid actions have -inf logits. Choosing random valid action.")
            valid_actions = np.where(board.flatten() == 0)[0]
            if len(valid_actions) == 0: # Truly no valid actions left
                 print("Error: No valid actions available, but game not ended.")
                 # Default to action 0, but this indicates an issue
                 action_idx = 0
                 # Create a dummy probability distribution for log_prob calculation
                 probs = torch.ones_like(logits) * (1.0 / self.num_actions)

            else:
                action_idx = random.choice(valid_actions)
                # Create a dummy probability distribution for log_prob calculation
                probs = torch.zeros_like(logits)
                probs[valid_actions] = 1.0 / len(valid_actions)


        else:
            probs = torch.softmax(masked_logits, dim=-1)
            try:
                action_dist = Categorical(probs)
                action_idx = action_dist.sample().item()
            except ValueError as e:
                 print(f"Error sampling action: {e}")
                 print("Logits:", logits)
                 print("Masked Logits:", masked_logits)
                 print("Probs:", probs)
                 # Fallback: Choose the highest probability valid action
                 valid_indices = torch.where(mask == 1)[0]
                 if len(valid_indices) > 0:
                    action_idx = valid_indices[torch.argmax(probs[valid_indices])].item()
                 else: # If somehow still no valid actions
                     action_idx = 0 # Default, indicates error


        # 5. Store log probability for training
        # Use the *original* probabilities before masking for sampling,
        # but calculate log_prob from the distribution used for sampling.
        log_prob = torch.log(probs[action_idx] + 1e-9) # Add epsilon for numerical stability
        self.log_probs.append(log_prob)

        return action_idx

    def update_policy(self):
        """Updates the policy using the REINFORCE algorithm."""
        if not self.rewards:
            return # Nothing to update

        self.model.train() # Set model to training mode

        # Calculate discounted returns
        returns = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        # Normalize returns (optional but often helpful)
        returns = torch.tensor(returns).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            # Avoid division by zero if only one step
             returns = (returns - returns.mean())


        # Calculate policy gradient loss
        policy_loss = []
        if len(self.log_probs) != len(returns):
             print(f"Warning: Mismatch in length of log_probs ({len(self.log_probs)}) and returns ({len(returns)}). Skipping update.")
             self.clear_buffers()
             return

        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R) # REINFORCE loss: -log_prob * G_t

        # Backpropagation
        self.optimizer.zero_grad()
        # Check if policy_loss is empty before summing
        if not policy_loss:
            print("Warning: Policy loss list is empty. Skipping backpropagation.")
            self.clear_buffers()
            return

        total_loss = torch.stack(policy_loss).sum()
        total_loss.backward()
        self.optimizer.step()

        # Clear buffers for the next episode
        self.clear_buffers()

    def clear_buffers(self):
        self.log_probs = []
        self.rewards = []

    def save_model(self, path):
        print(f"Saving model to {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        print(f"Loading model from {path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        
        
# --- Gomoku Environment (Provided in the prompt - slightly modified for compatibility) ---




# --- Training Loop ---
def train_llm_agent(env, agent, num_episodes=1000, print_interval=10, render_interval=100):
    episode_rewards = []
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        truncated = False # Gymnasium uses truncated
        total_reward = 0
        steps = 0

        while not done and not truncated:
            # Determine current player based on steps (or env.current_player if reliable)
            # Assuming player 1 starts (steps 0, 2, 4...)
            current_player = 1 if steps % 2 == 0 else -1
            
            # Ensure env player matches expectation (can remove if env manages it perfectly)
            if env.current_player != current_player:
                 print(f"Warning: Step count player ({current_player}) differs from env player ({env.current_player})")
                 current_player = env.current_player # Trust the environment


            # Agent selects action
            action = agent.select_action(state, current_player)

            # Environment steps
            next_state, reward, done, truncated, info = env.step(action)

            # Store reward (for the player who just moved)
            agent.rewards.append(reward)
            total_reward += reward # Track cumulative reward for info

            state = next_state
            steps += 1

            # Optional Rendering
            if render_interval > 0 and episode % render_interval == 0:
                 try:
                     env.render()
                     pygame.time.wait(50) # Small delay to see the move
                 except Exception as e:
                     print(f"Rendering failed: {e}")
                     # Consider disabling rendering if it keeps failing
                     # render_interval = 0

            # Break if error occurs (like invalid move termination)
            if "error" in info:
                 print(f"Episode {episode} ended early due to error: {info['error']} at step {steps}")
                 # The negative reward for invalid move is already in agent.rewards
                 break


        # --- End of Episode ---

        # Handle rewards for the *losing* player in REINFORCE
        # If player 1 won (steps is odd), the last reward was +1. The previous step's reward (player -1's last move) should reflect the loss.
        # If player -1 won (steps is even), the last reward was +1 (for player -1). Player 1's last move should reflect the loss.
        # A simple approach: if the game didn't end in a draw or error, the player whose turn *would* have been next lost.
        # We need to associate rewards with the correct player's actions.
        # REINFORCE updates based on the return *following* an action.
        # The current implementation stores reward received *after* taking an action.
        # If P1 takes action A1, receives R1, then P2 takes A2, receives R2 (and wins).
        # P1's trajectory: (S1, A1, R1), ...
        # P2's trajectory: (S2, A2, R2=Win_Reward)
        # This looks okay, the win/loss reward is associated with the final move.
        # The update uses discounted future rewards.

        # Update the policy using collected rewards and log_probs
        agent.update_policy()

        episode_rewards.append(total_reward)

        if episode % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            print(f"Episode: {episode}/{num_episodes}, Steps: {steps}, Total Reward: {total_reward:.2f}, Avg Reward (last {print_interval}): {avg_reward:.2f}")

    env.close() # Close pygame window
    return episode_rewards






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