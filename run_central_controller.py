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
from collections import defaultdict


from obstacle import MapGenerator
from constant import  CONSTANTS
from agent import Agent
from target import  Target
from env import Env
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
    
    memory = defaultdict(list)
    
    env= Env()
    controller = Cent_controller()
    agent_list, target_list = env.reset()
    # print(env.map)
    # print(env.validMap)
    env.render()
    for episode in range(CONST.LEN_EPISODE):
        action_list = controller.get_action(agent_list, target_list)
#        action_list = [getKeyPress(), 0]
#        print(action_list)
        agent_list, target_list = env.step(action_list)
        env.render()
        cv2.waitKey(20)
        
        ep_memory = []
#        ep_memory.append(env.map)
        ep_memory.append(env.agentList)
        ep_memory.append(env.targetList)
        ep_memory.append(action_list)
        
        memory[episode] = ep_memory
    
    # save memory
    pickle.dump( memory, open( "memory_100.p", "wb" ) )