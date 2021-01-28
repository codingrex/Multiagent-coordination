

import numpy as np
import random
import cv2
import time
import skimage.measure
import math



from obstacle import MapGenerator
from constant import  CONSTANTS

CONST= CONSTANTS()
mapGen = MapGenerator()

np.set_printoptions(precision=3, suppress=True)
class Env:
    def __init__(self):

        #make the map
        self.map= mapGen.generate_map(CONST.HEIGHT, CONST.WIDTH)

        self.validMap = np.zeros_like(self.map)
        self.height = self.map.shape[0]
        self.width = self.map.shape[1]
        self.agentList= None
        self.targetList = None





        #save video
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(f"checkpoints/cnn1.avi", self.fourcc, 50, (700, 700))





    
    def init_env(self, num_agents = 2):
        self.validMap= np.zeros_like(self.map)
        self.height= self.map.shape[0]
        self.width = self.map.shape[1]
        self.agentList= np.array([]).astype(int)
        self.targetList= np.array([]).astype(int)
        self.map= mapGen.generate_map(CONST.HEIGHT, CONST.WIDTH)
        self.rand_target_pos()
        self.rand_agents_pos()



    def rand_agents_pos(self, num_agents = 2):

        pos= self.random_pos(num_agents)

        self.agentList = np.append(self.agentList, pos)

        self.update_map()

    def rand_target_pos(self, num_target = 4):
        pos = self.random_pos(num_target)

        self.targetList= np.append(self.targetList, pos)

        self.update_map()

    def update_map(self):

        # agent are marked with negative numbers from -1...-n
        #in valid map: agent (-1)
        if len(self.agentList) != 0:
            agent_pos= self.agentList.reshape(-1, 2)
            a_index= -1
            for a_p in agent_pos:
                a_x= a_p[0]
                a_y= a_p[1]

                self.map[a_x, a_y]= a_index
                self.validMap[a_x, a_y] = -1
                a_index -= 1

        # targets are marked with positive numbers from 1...m
        # in valid map: target (1)
        if len(self.targetList) != 0:
            target_pos = self.targetList.reshape(-1, 2)
            t_index = 1
            for t_p in target_pos:
                t_x = t_p[0]
                t_y = t_p[1]
                self.map[t_x, t_y] = t_index
                self.validMap[t_x, t_y] = 1
                t_index += 1






    def random_pos(self, num):

        res= []

        valid_map= self.validMap.copy()


        for i in range(num):
            x = np.random.randint(0, self.height)
            y = np.random.randint(0, self.width)
            pos = [x, y]

            while not self.is_valid_location(pos, valid_map):
                x = np.random.randint(0, self.height)
                y = np.random.randint(0, self.width)
                pos = [x, y]


            res += pos
            valid_map[x, y] = 1

        return res



    def view_agents_pos(self):
        pos= np.array(self.agentList)
        pos= pos.reshape(-1,2)
        return pos

    def view_targets_pos(self):
        pos= np.array(self.targetList)
        pos= pos.reshape(-1,2)
        return pos




    #mode 0 is for generating random pos, mode 1 is for movement
    def is_valid_location(self, pos, valid_map, mode= 0):
        # range check
        if 0 <= pos[0] < valid_map.shape[0] and 0<= pos[1] < valid_map.shape[1]:
            pass
        else:
            return False
        # availability check
        if mode == 0:
            if valid_map[pos[0], pos[1]] == 0:
                return True
            else:
                return False
        else:
            if valid_map[pos[0], pos[1]] == -1:
                return False
            else:
                return True




    def step_agents(self, actionList):
        movList =np.array([env.interprete_action(i) for i in actionList])
        index= 0

        poslist = env.view_agents_pos()

        for p, t in zip(poslist, movList):
            new_p = p + t
            if self.is_valid_location(new_p, self.validMap, 1):
                self.agentList[index] = new_p[0]
                self.agentList[index + 1] = new_p[1]
                # after movement, clear past pos
                self.clear_pos(p)
            index += 2

        poslist = env.view_agents_pos()
        indices = np.array([]).astype(int)

        # if agent gets the target
        for p in poslist:
            x= p[0]
            y= p[1]
            # if new pos is on a target in the map
            if self.map[x, y] > 0:
                indices= np.append(indices, [x, y])
                self.clear_targets(self.map[x, y])

        #remove targets in the target list, can be simultaneous
        self.targetList = np.delete(self.targetList, indices)


        #update map at last

        self.update_map()


    #clear target object on map:
    def clear_targets(self, index):
        pos= int (2 * (index - 1))

        self.clear_pos([self.targetList[pos], self.targetList[pos + 1]])





    #clear the past position
    def clear_pos(self, pos):
        x= pos[0]
        y= pos[1]

        self.map[x, y]= 0
        self.validMap[x, y]= 0








    #interprete action as movement
    def interprete_action(self, action):
        #0- stay
        if action == 0:
            return [0,0]
        #1- down
        elif action == 1:
            return [1, 0]
        #2 -up
        elif action == 2:
            return [-1, 0]
        #3- left
        elif action == 3:
            return [0, -1]
        #4- right
        elif action == 4:
            return [0, 1]



    
    def get_action_space(self):
        return [0,1,2,3,4]


    def step(self, action_list):
        env.step_agents(action_list)
        #can add a pattern to generate targets here!
    


    def reset(self):
        
        self.init_env()
    
    def render_prep(self, map):
        mapshow = np.rot90(map, 1)

        mapshow = np.where(mapshow > 0, 255, mapshow)
        mapshow = np.where(mapshow < 0, 150, mapshow)


        mapshow = mapshow.astype(np.uint8)

        mapshow = cv2.applyColorMap(mapshow, cv2.COLORMAP_JET)

        return mapshow

    def render(self):

        img = np.copy(self.map)


        """ initialize heatmap """

        full_map = self.render_prep(img)
        full_map = cv2.resize(full_map,(700,700),interpolation = cv2.INTER_AREA)



        cv2.imshow("Map", full_map)


        cv2.waitKey()
#
#     def get_reward(self, current_map):
#
#         #sum up reward on all free pixels
#         actualR = np.where((current_map<= 0), current_map, 0)
#         curSumR = np.sum(actualR)
#
#
#         return curSumR
#
#     def get_reward_local(self, local_map_list, current_map):
#         local_reward_list = []
#         #sum up reward on all free pixels
#         for local_map in local_map_list:
#             actualR = np.where((local_map<= 0), local_map, 0)
#             curSumR = np.sum(actualR)
#             local_reward_list.append(curSumR)
#         sharedR = np.where((current_map<= 0), current_map, 0)
#         shared_reward = np.sum(sharedR)
# #        print(local_reward_list, shared_reward)
#         return local_reward_list, shared_reward
#
#     def get_local_heatmap_list(self, current_map, agent_g_pos_list):
#         local_heatmap_list = []
#         for g in agent_g_pos_list:
#             r = int((CONST.LOCAL_SZ -1) /2)
#             lx = int(max(0, g[1] - r))
#             hx = int(min(CONST.MAP_SIZE, r + g[1] + 1))
#             ly = int(max(0, g[0] - r))
#             hy = int(min(CONST.MAP_SIZE, r + g[0] + 1))
#             tempMask = np.zeros_like(current_map)
#             tempMask[lx: hx , ly : hy] = 1
#
#             local_view = np.ones((CONST.LOCAL_SZ,CONST.LOCAL_SZ)) * 150
#
#             llx = int(lx - (g[1] - r))
#             hhx = int(hx - g[1] + r)
#
#             lly = int(ly - (g[0] - r))
#             hhy = int(hy - g[0] + r)
#
#             local_view[llx: hhx, lly: hhy] = current_map.T[lx: hx , ly : hy]
#             local_heatmap_list.append(local_view.T)
#         return local_heatmap_list
#
#     def get_mini_map(self, current_map, ratio, agent_g_pos):
#         num_windows = int(current_map.shape[0] * ratio)
#
#
#         window_sz = int(1/ratio)
#
#         mini_obs = cv2.resize(self.obstacle_map,(num_windows,num_windows),interpolation = cv2.INTER_AREA)
#         mini_obs = np.where(mini_obs > 0, 150, 0)
#
#         decay_map = np.where(current_map < 0, current_map, 0)
#         mini_decay = skimage.measure.block_reduce(decay_map, (window_sz,window_sz), np.min)
#
#         mini_heatmap = np.where(mini_decay < 0, mini_decay, mini_obs)
#
#
#         """
#         for gpos in agent_g_pos:
#             mini_heatmap[int(gpos[0] * ratio), int(gpos[1] * ratio)] = 100
#         """
#
#
#         agent_minimap_list = []
#         for gpos in agent_g_pos:
#             agent_minimap = np.copy(mini_heatmap)
#             agent_minimap[int(gpos[0] * ratio), int(gpos[1] * ratio)] = 200
#             agent_minimap_list.append(agent_minimap)
#         return agent_minimap_list
#
#     def save2Vid(self, episode, step):
#
#         img = np.copy(self.current_map_state)
#
#         reward_map = img
#
#         """ initialize heatmap """
#
#         full_heatmap = self.heatmap_render_prep(reward_map)
#         full_heatmap = cv2.resize(full_heatmap, (700, 700), interpolation=cv2.INTER_AREA)
#         display_string = "Episode: " + str(episode) + " Step: " + str(step)
#         full_heatmap = cv2.putText(full_heatmap, display_string, (20,20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,255,255) , 2, cv2.LINE_AA)
#         self.out.write(full_heatmap.astype('uint8'))


if __name__ == '__main__':
    env= Env()
    env.init_env()

    # manually spawn agents and targets
    # env.agentList= np.array([0,0, 2,2])
    # env.targetList = np.array([0, 1, 2, 3])
    env.update_map()
    print(env.map)
    print(env.validMap)

    # env.step_agents([4,4])
    # env.step_agents([1, 1])
    # env.step_agents([4, 4])
    # env.step_agents([4, 0])
    # env.step_agents([1, 0])

    # env.step([4, 4])
    # env.step([1, 1])
    # env.step([4, 4])


    # print(env.map)
    # print(env.validMap)

    env.render()




    # env.rand_agents_pos()
    # # env.rand_target_pos()
    # print(env.agentList)
    # # print(env.targetList)
    # print(env.validMap)
    # print(env.map)
    #
    #
    #
    #
    #
    #
    # env.step_agents([0,1])
    #
    # print(env.agentList)
    # # print(env.targetList)
    # print(env.validMap)
    # print(env.map)

    # print(env.interprete_action(1))
    # actionList= [0,1,2,3]
    #
    # print(np.append(actionList, [4,5]))

    # movList =np.array([env.interprete_action(i) for i in actionList]).flatten()
    #
    #
    # testarr1= np.array([1,2,3,4])
    # testarr2 = np.array([1, 2, 3, 4])
    #
    # print(testarr1 + testarr2)




        