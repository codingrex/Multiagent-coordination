import numpy as np


class CONSTANTS:
    def __init__(self):
        self.LEN_EPISODE = 800
        self.HEIGHT= 10
        self.WIDTH= 10
        self.START_TIME= 10
        self.NUMBER_AGENTS= 2
        self.EPSILON_TARGET = 0.2
        self.MAX_TARGETS = int(self.HEIGHT * self.WIDTH / 0.3)
        
        self.INIT_NUM_TARGETS = 5
        
        # Training
        self.NUM_DATA_POINTS = 1000
        self.UNCERTAINITY = False
