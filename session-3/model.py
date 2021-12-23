import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3,stride=1,padding=1)
        self.drop = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(64*15*15, 1000)
        self.fc2 = nn.Linear(1000, 1)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = x.view([64*15*15,-1])
        x = F.relu(self.fc1(x))
        y = F.sigmoid(self.fc2(x))
        return y
