import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


# hungarian assignment
cost = np.array([[4, 1, 3], [2, 0, 5]])
row_ind, col_ind = linear_sum_assignment(cost)
# [agent 1, agent2] -> columns (targets): [t1, t2]
print("assignment")
print(col_ind)
# [agent 1, agent2] -> optimal cost [cost_t1, cost_t2]
print("optimal cost")
print(cost[row_ind, col_ind].sum())


# confusion matrix
y_true = [2, 0, 2, 2, 0, 1,3]
y_pred = [0, 0, 2, 2, 0, 2,3]
print("Confusion matrix")
print(confusion_matrix(y_true, y_pred))


import pickle
import numpy as np
from supv_agent import SupvNet
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

test_memory = pickle.load( open( "memory_test.p", "rb" ) )
net = SupvNet()
net.load_model("predictor_10000.model")

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

conf_matrix = confusion_matrix(labels.numpy(), pred_actions.numpy())
print(conf_matrix)
