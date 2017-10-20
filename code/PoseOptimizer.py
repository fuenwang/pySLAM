import os
import sys
import cv2
import numpy as np
import scipy
from scipy.optimize import least_squares
import Dataset
import Cameras

class PoseOptimizer:
    def __init__(self):
        pass
    
    def TwoViewPose(self, img1, depth1, img2):
    #   Calculate two-view pose
        depth_mask1 = (depth1 != -1)
        depth_mask2 = (depth2 != -1)


if __name__ == '__main__':
    data = Dataset.Dataset('../data')

    [img1, depth1] = data[2]
    [img2, depth2] = data[5]

