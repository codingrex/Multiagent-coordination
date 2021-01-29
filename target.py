import numpy as np
import random

from constant import  CONSTANTS

CONST= CONSTANTS()


class Target:
    def __init__(self, pos, time= CONST.START_TIME):
        self.pos = pos
        self.time= time

    def decay(self):
        self.time -= 1


