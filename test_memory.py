# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:21:30 2021

@author: amris
"""

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



net = SupvNet()
net.load_model("predictor_10000.model")

#summary(net, (3, 10, 10))



test_memory = pickle.load( open( "memory_test.p", "rb" ) )
test_states, test_actions = torch.from_numpy(np.array(test_memory[0]),), torch.from_numpy(np.array(test_memory[1]))

# get the inputs; data is a list of [inputs, labels]
inputs, labels = test_states, test_actions.long()


# zero the parameter gradients
#    optimizer.zero_grad()

# forward + backward + optimize
outputs = net(inputs.float())
#    loss.backward()
#    optimizer.step()

pred_actions = outputs.argmax(axis = 1)

accuracy = pred_actions == labels
accuracy = accuracy.numpy()

accuracy = np.sum(accuracy)/ len(accuracy)

print(accuracy)


