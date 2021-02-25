# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:24:57 2021

@author: amris
"""

import numpy as np
import random
import cv2
import time
import skimage.measure
import math
import keyboard
import pickle
from tqdm import tqdm

from obstacle import MapGenerator
from constant import  CONSTANTS
from agent import Agent
from target import  Target
from env import Env
#from linear_sum_assignment_controller import Cent_controller
from centralized_controller import Cent_controller

CONST= CONSTANTS()

def getKeyPress():
    act = 0
    key_press = keyboard.read_key()
    if key_press == "up":
        act = 4
    elif key_press == "left":
        act = 2
    elif key_press == "down":
        act = 3
    elif key_press == "right":
        act = 1
    else:
        act = 0
    return act

if __name__ == '__main__':
    
    memory_state = []
    memory_action = []
    memory_num_targets = []
    
    env= Env()
    controller = Cent_controller()
    agent_list, target_list = env.reset()
    # print(env.map)
    # print(env.validMap)
    env.render()
    for episode in tqdm(range(CONST.LEN_EPISODE)):
        current_map = np.copy(env.map)

        action_list = controller.get_action(agent_list, target_list)
#        action_list = [getKeyPress(), 0]
#        print(action_list)
        
        
        agent_list, target_list = env.step(action_list)
#        env.render()
#        cv2.waitKey(1)
        if CONST.UNCERTAINITY:
            a1_map = np.where(current_map == -1, 1, 0)
            a2_map = np.where(env.get_quadMap(0, 2, 2) == -2, 1, 0)
            target_map = np.where(current_map > 0, current_map, 0)
            
    #        # 1 hot encode the actions
    #        a1_action = np.zeros(5)
    #        a1_action[action_list[0]] = 1
    #        a2_action = np.zeros(5)
    #        a2_action[action_list[1]] = 1
            
            # use action as value
            a1_action = action_list[0]
            a2_action = action_list[1]
            memory_num_targets.append(np.count_nonzero(target_map > 0))
            memory_num_targets.append(np.count_nonzero(target_map > 0))
            
            # agent 1
            learn_map = np.array([a1_map, a2_map, target_map])
            memory_state.append(learn_map)
            memory_action.append(a2_action)
            
            # agent 2
            a1_map = np.where(env.get_quadMap(1, 2, 2) == -2, 1, 0)
            a2_map = np.where(current_map == -2, 1, 0)
            target_map = np.where(current_map > 0, current_map, 0)
    
            learn_map = np.array([a2_map, a1_map, target_map])
            memory_state.append(learn_map)
            memory_action.append(a1_action)
        else:
            a1_map = np.where(current_map == -1, 1, 0)
            a2_map = np.where(current_map == -2, 1, 0)
            target_map = np.where(current_map > 0, current_map, 0)
            
    #        # 1 hot encode the actions
    #        a1_action = np.zeros(5)
    #        a1_action[action_list[0]] = 1
    #        a2_action = np.zeros(5)
    #        a2_action[action_list[1]] = 1
            
            # use action as value
            a1_action = action_list[0]
            a2_action = action_list[1]
            memory_num_targets.append(np.count_nonzero(target_map > 0))
            memory_num_targets.append(np.count_nonzero(target_map > 0))
            
            # agent 1
            learn_map = np.array([a1_map, a2_map, target_map])
            memory_state.append(learn_map)
            memory_action.append(a2_action)
            
            # agent 2
            a1_map = np.where(current_map == -1, 1, 0)
            a2_map = np.where(current_map == -2, 1, 0)
            target_map = np.where(current_map > 0, current_map, 0)
    
            learn_map = np.array([a2_map, a1_map, target_map])
            memory_state.append(learn_map)
            memory_action.append(a1_action)

    # To balance data with action 0 with and without targets
    avg_1234 = 0
    for a in [1,2,3,4]:
        occ = np.where(np.array(memory_action) == a)[0].shape[0]
        avg_1234 += occ
    avg_1234 = int(avg_1234/4)

    action_1234_idx = np.where(np.array(memory_action) > 0)[0].tolist()

    target_act0_mem = np.copy(np.array(memory_num_targets))
    target_act0_mem[action_1234_idx] = -1

    target0_act0_idx = np.where(np.array(target_act0_mem) == 0)[0]
    targetn0_act0_idx = np.where(np.array(target_act0_mem) > 0)[0]

    print(avg_1234, target0_act0_idx.shape, targetn0_act0_idx.shape)

    remove_idx_target0 = np.random.choice(target0_act0_idx, len(target0_act0_idx) - avg_1234, replace=False)
    remove_idx_targetn0 = np.random.choice(targetn0_act0_idx, len(targetn0_act0_idx) - avg_1234, replace=False)

    all_remove = np.concatenate((remove_idx_target0, remove_idx_targetn0))
    
    for index in sorted(all_remove, reverse=True):
        del memory_action[index]
        del memory_state[index]
        del memory_num_targets[index]
        
#    memory_action = np.delete(np.array(memory_action), all_remove.tolist())
#    memory_state = np.delete(np.array(memory_state), all_remove.tolist())
#    memory_num_targets = np.delete(np.array(memory_num_targets), all_remove.tolist())

    # save memory
    print("Data points before filterin = ", len(memory_state))
    remove_excess = len(memory_state) - CONST.NUM_DATA_POINTS
    if remove_excess < 0:
        print("Memory less than length of episode")
    else:
        remove_idx = np.random.choice(np.array(range(len(memory_state))), remove_excess, replace = False)
        for index in sorted(remove_idx, reverse=True):
            del memory_action[index]
            del memory_state[index]
            del memory_num_targets[index]
    print("Data points = ", len(memory_state))
#    pickle.dump( [memory_state, memory_action], open( "data/certain_test.p", "wb" ) )
    
    
    
    
    
    