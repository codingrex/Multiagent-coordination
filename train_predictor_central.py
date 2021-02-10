# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 19:53:44 2021

@author: amris
"""

import pickle
import numpy as np
from supv_agent import SupvNet
import torch.nn as nn
import torch.optim as optim
import torch

memory = pickle.load( open( "memory_100.p", "rb" ) )


net = SupvNet()

#summary(net, (3, 10, 10))


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr= 0.5, momentum=0.9)

for epoch in range(400):  # loop over the dataset multiple times

    memory_states, memory_actions = torch.from_numpy(np.array(memory[0]),), torch.from_numpy(np.array(memory[1]))



    running_loss = 0.0
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = memory_states, memory_actions.long()


    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs.float())
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    print('[%d] loss: %.3f' %
          (epoch + 1, running_loss))
    running_loss = 0.0

print('Finished Training')