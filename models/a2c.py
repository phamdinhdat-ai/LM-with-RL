import torch
import torch.nn as nn
from torch.nn import functional as F
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        
        # Actor
        self.actor_fc = nn.Linear(128, output_dim)
        
        # Critic
        self.critic_fc = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.actor_fc(x)  # Policy logits
        value = self.critic_fc(x)  # State-value estimate
        return logits, value
