import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOSubAgent(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PPOSubAgent, self).__init__()
        # Shared feature extractor
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim))
            
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        
    def forward(self, x):
        return self.actor(x), self.critic(x)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))