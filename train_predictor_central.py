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
import matplotlib.pyplot as plt

memory = pickle.load( open( "memory_a1_10000.p", "rb" ) )
test_memory = pickle.load( open( "memory_test.p", "rb" ) )


net = SupvNet()
#net.load_model("predictor_10000.model")

#summary(net, (3, 10, 10))

loss_arr = []
accuracy_arr = []

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr= 0.1, momentum=0.9)

for epoch in range(1500):  # loop over the dataset multiple times

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
    
    loss_arr.append(running_loss)
    running_loss = 0.0
    
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
    accuracy_arr.append(accuracy)
    
    if epoch % 100 == 10:
        net.save_model("predictor_10000.model")
        plt.close()
        plt.plot(loss_arr)
        plt.plot(accuracy_arr)
        plt.show()
        plt.pause(0.1)
        
    
    
print('Finished Training')
net.save_model("predictor_10000.model")
