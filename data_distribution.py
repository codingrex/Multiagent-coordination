# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:21:30 2021

@author: amris
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


training_memory = pickle.load(open( "data/uncertain_train.p", "rb" ))

def plot_3d_hist(fig_num, title, distribution):
    
    # 3d histogram
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')
   
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(np.array(range(10))+ 0.25, np.array(range(10)) + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = distribution.ravel()
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(title)
    plt.show()

def agent_target_distribution(states):
    mem_state = np.array(states)
    
    agent1_distribution = np.sum(mem_state[::2,0,:,:], axis = 0)
    agent2_distribution = np.sum(mem_state[::2,1,:,:], axis = 0)
    
    target_maps = []
    num_targets = []
    for target_map in mem_state[:,2,:,:]:
        target_map = np.where(target_map > 0, 1, 0)
        num_targets.append(np.count_nonzero(target_map == 1))
        target_maps.append(target_map)

    target_distribution = np.sum(np.array(target_maps), axis = 0)
    
    plt.figure(1)
    n, bins, patches = plt.hist(x=num_targets, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Number of targets at a time step')
    plt.ylabel('Frequency')
    plt.title('Number of targets')
    
    plot_3d_hist(2, 'Agent 1 position distribution', agent1_distribution)
    plot_3d_hist(3, 'Agent 2 position distribution', agent2_distribution)
    plot_3d_hist(4, 'target position distribution', target_distribution)
    
    print(np.sum(agent1_distribution))
    print(np.sum(agent2_distribution))
    print(target_distribution)
    

def action_distribution(num_fig, title, distribution):
    
    plt.figure(num_fig)
    n, bins, patches = plt.hist(x=distribution, bins='auto', color='#0504aa',
                            label = ['0_no_targets','0', '1', '2', '3', '4'],alpha=1.0, rwidth=1.0, align='mid')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.title(title)
    
agent_target_distribution(training_memory[0])

action_distribution(5, "Agent 1 action distribution", np.array(training_memory[1])[::2])
action_distribution(6, "Agent 2 action distribution", np.array(training_memory[1])[1::2])
