import torch
import torch.nn as nn

class RegressionModel(nn.Module):
    # You should build your model with at least 2 layers using tanh activation in between
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fl1 = nn.Linear(input_size,hidden_size)
        self.fl2 = nn.Linear(hidden_size,1)

    def forward(self, x):
        x = nn.functional.tanh(self.fl1(x))
        y = self.fl2(x)
        return y