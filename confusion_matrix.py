import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


import pickle
import numpy as np
from supv_agent import SupvNet
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

#model = "old_cntrlr_trained.model"
#test_memory_filename = "test_old_controller.p"


model = "predictor_10000.model"
test_memory_filename = "memory_test.p"

test_memory = pickle.load( open( test_memory_filename, "rb" ) )
net = SupvNet()
net.load_model(model)

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
