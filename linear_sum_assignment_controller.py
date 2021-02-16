# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:10:45 2021

@author: amris
"""
import numpy as np

from scipy.optimize import linear_sum_assignment as lsa

class Cent_controller:
    def __init__(self):
        pass
    def get_action(self, agent_list, target_list):
        agent_pos_list = np.array([agent.pos for agent in agent_list])
        target_pos_list = np.array([target.pos for target in target_list])
        target_time_list = np.array([target.time for target in target_list])
        
        if len(target_pos_list) == 0:
            return [0]*len(agent_pos_list)
        
        agent2target_list = []
        
        for i, agent_pos in enumerate(agent_pos_list):
            agent2target_list.append([])
            for target_pos, target_time in zip(target_pos_list, target_time_list):
                distVtime = np.sum(np.abs(target_pos - agent_pos))
                agent2target_list[i].append(distVtime)
        
        target_for_agent = []
        
        
        cost = np.array(agent2target_list)
        
        if len(target_list) >= 2:
            _, target_for_agent = lsa(cost)
        elif 0 < len(target_list) < len(agent_pos_list):
            cost = np.hstack((cost, 100 * np.ones((len(agent_pos_list),1))))
            _, target_for_agent = lsa(cost)
        else:
            target_for_agent = [-1]*len(agent_pos_list)
        
        action_list = []
        for agent_idx, target_idx in enumerate(target_for_agent):
            if (target_idx == -1 or 
                target_idx > len(target_list)-1 or
                np.sum(np.abs(target_pos_list[target_idx] - agent_pos_list[agent_idx])) > target_time_list[target_idx]):
                action_list.append(0)
            else:
                x,y = target_pos_list[target_idx] - agent_pos_list[agent_idx]
                
                if x == 0 and y == 0:
                    action_list.append(0)
                    
                elif x == 0 or y == 0:
                    if not x == 0:
                        if x > 0:
                            action_list.append(1)
                        else:
                            action_list.append(2)
                    elif not y == 0:
                        if y > 0:
                            action_list.append(4)
                        else:
                            action_list.append(3)
                else: # both are non zero
                    choice = np.random.randint(0,2)
                    if choice == 0:
                        if x > 0:
                            action_list.append(1)
                        else:
                            action_list.append(2)
                    else:
                        if y > 0:
                            action_list.append(4)
                        else:
                            action_list.append(3)
            
        return action_list