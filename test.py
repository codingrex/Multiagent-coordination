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
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print("Confusion matrix")
print(confusion_matrix(y_true, y_pred))
