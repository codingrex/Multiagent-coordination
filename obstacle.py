# -*- coding: utf-8 -*-

import skgeom as sg
import numpy as np
from matplotlib.path import Path
from collections import defaultdict
from functools import partial

import time


class MapGenerator:
    def __init__(self):
        pass


    def generate_map(self, height, width):
        return np.zeros((height, width))

    def add_obstacle(self, map, seed):

        # wait to be completed
        return map




# # Testing function
# if __name__ == '__main__':
#     map= MapGenerator().generate_map(3,2)
#     map = MapGenerator().add_obstacle(map, 0)
#
#
#     print(map)
#     print(map.shape)




    
