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


from obstacle import MapGenerator
from constant import  CONSTANTS
from agent import Agent
from target import  Target
from env import Env
from linear_sum_assignment_controller import Cent_controller

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
    
    env= Env()
    controller = Cent_controller()
    agent_list, target_list = env.reset()
    # print(env.map)
    # print(env.validMap)
    env.render()
    for episode in range(CONST.LEN_EPISODE):
        print(episode)
        current_map = np.copy(env.map)

        action_list = controller.get_action(agent_list, target_list)
#        action_list = [getKeyPress(), 0]
#        print(action_list)
        
        
        agent_list, target_list = env.step(action_list)
#        env.render()
#        cv2.waitKey(1)
        
        
        a1_map = np.where(current_map == -1, 1, 0)
        a2_map = np.where(current_map == -2, 1, 0)
        target_map = np.where(current_map > 0, env.map, 0)
        
#        # 1 hot encode the actions
#        a1_action = np.zeros(5)
#        a1_action[action_list[0]] = 1
#        a2_action = np.zeros(5)
#        a2_action[action_list[1]] = 1
        
        # use action as value
        a1_action = action_list[0]
        a2_action = action_list[1]
        
        # agent 1
        learn_map = np.array([a1_map, a2_map, target_map])
        memory_state.append(learn_map)
        memory_action.append(a2_action)
        
        # agent 2
        learn_map = np.array([a2_map, a1_map, target_map])
        memory_state.append(learn_map)
        memory_action.append(a1_action)
    
    # save memory
    print(len(memory_state))
#    pickle.dump( [memory_state, memory_action], open( "memory_test.p", "wb" ) )
    
    
    
    
    
    