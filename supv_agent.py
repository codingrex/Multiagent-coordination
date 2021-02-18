import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torch.distributions import Categorical
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
import random

    class SupvNet(nn.Module):
    def __init__(self):
        super(SupvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()
    
    
#net = SupvNet()
#
#summary(net, (3, 10, 10))
#
#
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)