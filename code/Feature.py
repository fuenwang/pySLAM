import cv2
import vlfeat
import pyflann
import numpy as np

def Extract(config, img):
    height = img.shape[0]
    width = img.shape[1]
    peak = config.Get('sift_peak_threshold')
    edge = config.Get('sift_edge_threshold')
    loc, des = vlfeat.vl_sift(img, peak_thresh=peak, edge_thresh=edge)
    #loc, des = vlfeat.vl_sift(img)
    loc = np.round(loc[0:2, :].T)
    loc[:, 0] -= width / 2
    loc[:, 1] -= height / 2
    des = des.T
    return loc, des.astype(np.float)

def Match(config, frame1, frame2):
    [loc1_all, des1] = Extract(config, frame1)
    [loc2_all, des2] = Extract(config, frame2)

    flann = pyflann.FLANN()
    result, dist = flann.nn(des2, des1, 2, algorithm="kmeans", branching=32, iterations=10, checks=200)

    index1 = np.arange(loc1_all.shape[0])
    dist[dist == 0] = 0.000001
    compare = (dist[:, 0].astype(np.float32) / dist[:, 1]) < config.Get('flann_threshold')
    index1 = index1[compare]
    index2 = result[:, 0][compare]
    #index2 = result[:, 0]

    loc1 = loc1_all[index1, :]
    loc2 = loc2_all[index2, :]
    [F, M] = cv2.findFundamentalMat(loc1, loc2, cv2.FM_RANSAC)
    M = np.reshape(M, [-1])
    index1 = index1[M == 1]
    index2 = index2[M == 1]
    #print loc1.shape[0], index1.shape
    #print index1.shape
    return np.vstack([index1, index2]).T, loc1_all, loc2_all

class Feature:
    def __init__(self):
        self._data = {}

    def Add(self, image_name, loc):
        self._data[image_name] = loc

    def Get(self, image_name, index = None):
        if index is None:
            return self._data[image_name]
        else:
            return self._data[image_name][index, :]


