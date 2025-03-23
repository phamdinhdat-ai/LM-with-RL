# from envs.gomoku_env import GomokuEnv
from envs.gomoku_env import GomokuEnv
# 
def test_env():
    env = GomokuEnv(rows=15, cols=15, win_length=5)
    
    obs, _ = env.reset()
    assert obs.shape == (15, 15), "Observation shape mismatch"
    
    action = 0  # Place at top-left corner
    obs, reward, done, _, _ = env.step(action)
    assert obs[0, 0] == 1, "Player 1 move failed"
    
    action = 1  # Player 2 moves
    obs, reward, done, _, _ = env.step(action)
    assert obs[0, 1] == -1, "Player 2 move failed"
    
    print("Environment basic tests passed!")

if __name__ == "__main__":
    test_env()