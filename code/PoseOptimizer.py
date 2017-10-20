import os
import sys
import cv2
import numpy as np
import scipy
from scipy.optimize import least_squares
import Dataset
import Cameras

#CameraModel = Cameras.CameraModel()

def PhotometricError(pose, points, img1, img2, mask, obj, h, w):
    points_2D = obj.PerspectiveProject(points, pose)
    
    img1_flat = img1.reshape([h * w])
    #img2_flat = img2.reshape([h * w])
    return np.abs(img1_flat[mask] - (img2[points_2D[:, 1], points_2D[:, 0]].reshape([h * w]))[mask])

class PoseOptimizer:
    def __init__(self, height, width):
        self.h = height
        self.w = width
        self.camera = Cameras.CameraModel(height, width)
    
    def TwoViewPose(self, img1, depth1, img2):
    #   Calculate two-view pose
        h = self.h
        w = self.w
        model = self.camera
        #img1_flat = img1.reshape([h * w])
        #img2_flat = img2.reshape([h * w])
        depth_mask = (depth1 != -1).reshape([h*w])

        points_3D = model.PointFromDepth(depth)

        
if __name__ == '__main__':
    data = Dataset.Dataset('../data')

    [img1, depth1] = data[2]
    [img2, depth2] = data[5]
    print img2.shape
    print depth2.shape
