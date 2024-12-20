import torch
from torch import nn

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 8)
        self.fc3 = nn.Linear(8, output_size)
    
    def forward(self, state):
        x = torch.nn.functional.gelu(self.fc1(state))
        x = torch.nn.functional.gelu(self.fc2(x))
        return self.fc3(x)