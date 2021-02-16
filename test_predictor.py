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
import torch



from obstacle import MapGenerator
from constant import  CONSTANTS
from agent import Agent
from target import  Target
from env import Env
from centralized_controller import Cent_controller
from supv_agent import SupvNet

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
    
    predictor = SupvNet()
    predictor.load_model("predictor_10000.model")
    
    memory_state = []
    num_correct_predictions = 0    
    
    env= Env()
    controller = Cent_controller()
    agent_list, target_list = env.reset()
    # print(env.map)
    # print(env.validMap)
    env.render()
    for episode in range(CONST.LEN_EPISODE):
        current_map = np.copy(env.map)

        controller_action_list = controller.get_action(agent_list, target_list)
#        action_list = [getKeyPress(), 0]
#        print(action_list)
        
        
        agent_list, target_list = env.step(controller_action_list)
        env.render()
        cv2.waitKey(20)
        
        
        a1_map = np.where(current_map == -1, 1, 0)
        a2_map = np.where(current_map == -2, 1, 0)
        target_map = np.where(current_map > 0, env.map, 0)
        
#        # 1 hot encode the actions
#        a1_action = np.zeros(5)
#        a1_action[action_list[0]] = 1
#        a2_action = np.zeros(5)
#        a2_action[action_list[1]] = 1
        
        # use action as value
        
        # agent 1
        agent1_map = np.array([a1_map, a2_map, target_map])
        
        # agent 2
        agent2_map = np.array([a2_map, a1_map, target_map])
    
        # compare controller and predictor action list
        predictor_action_list = predictor(torch.from_numpy(np.array([agent2_map, agent1_map]),).float())
        
        for controller, predictor in zip(controller_action_list, predictor_action_list):
            if controller == predictor.argmax():
                num_correct_predictions += 1
        
    # check percentage
    print("Prediction Percentage = ", 100* num_correct_predictions/(2*CONST.LEN_EPISODE))
        
    
    
    
    
    