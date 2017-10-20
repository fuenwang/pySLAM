import cv2
import numpy as np
import camera_params as ca

intrisic_rgb = np.array([
        [ca.fx_rgb, 0, ca.cx_rgb],
        [0, ca.fy_rgb, ca.cy_rgb],
        [0, 0, 1]
    ])
#print intrisic_rgb

def ConcatPose(R, T):
    out = np.zeros([3, 4], np.float64)
    out[:, :3] = R
    out[:, 3] = T
    return out

class CameraModel:
    def __init__(self, height, width):
        self.h = height
        self.w = width
        self.x = np.tile(np.arange(0, width), [height, 1])
        self.y = np.tile(np.arange(0, height), [width, 1]).T

    def PointFromDepth(self, depth):
        # Covert depth map to 3d points
        h = self.h
        w = self.w
        x = self.x.reshape([h*w])
        y = self.y.reshape([h*w])
        depth_flat = depth.reshape([h * w])
        points = np.zeros([h * w, 3], np.float64)

        points[:, 0] = (x - ca.cx_rgb) / ca.fy_rgb
        points[:, 1] = (y - ca.cy_rgb) / ca.fx_rgb
        points[:, 2] = ca.fx_rgb * ca.fy_rgb

        norm = np.linalg.norm(points, axis = 1)

        for i in range(3):
            points[:, i] = points[:, i] / norm * depth_flat

        return points.reshape([h, w, 3])

    def PerspectiveProject(self, points, pose):
        # Project 3d points into image plane
        # pose[0:3] is rotation, pose[3:] is translation
        h = self.h
        w = self.w
    
        points_flat = points.reshape([h * w, 3])

        R = cv2.Rodrigues(pose[:3])
        M = ConcatPose(R, pose[3:])

        project_matrix = np.dot(intrisic_rgb, M)


        homo_product = np.dot(project_matrix, points_flat.T).T
        for i in range(2):
            homo_product[:, i] /= homo_product[:, 2]

        return np.round(homo_product[:, :2]).astype(np.int32)







