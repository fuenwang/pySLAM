import os
import sys
import cv2
import Data
import Config
import Feature
import pyopengv
import numpy as np

def Init(config, dataset):
    good_frame = False
    init_frame_name, init_frame = dataset.Next()

    [loc, des] = Feature.Extract(config, init_frame)
    while not good_frame:
        frame_name, frame = dataset.Next()
        #print frame_name
        #print loc.shape
        [match, _, _] = Feature.Match(config, init_frame, frame)
        if match.shape[0] / float(loc.shape[0]) < config.Get('key_frame_threshold'):
            good_frame = True

    return init_frame_name, frame_name

def Run(config, dataset, frame1, frame2):
    Feature_bag = Feature.Feature()
    key_frame = [frame1, frame2]

    img1 = dataset.ReadImg(frame1)
    img2 = dataset.ReadImg(frame2)

    [match, loc1, loc2] = Feature.Match(config, img1, img2)
    Feature_bag.Add(frame1, loc1)
    Feature_bag.Add(frame2, loc2)
    
    match_pt1 = Feature_bag.Get(frame1, index = match[:, 0])
    match_pt2 = Feature_bag.Get(frame2, index = match[:, 1])

    a = np.ones([match_pt1.shape[0], 3], float)
    b = np.ones([match_pt2.shape[0], 3], float)

    a[:, 0:2] = match_pt1
    b[:, 0:2] = match_pt2
    a[:, :] /= np.linalg.norm(a, axis=1)[:, np.newaxis]
    b[:, :] /= np.linalg.norm(b, axis=1)[:, np.newaxis]
    #print a
    print b
    M = pyopengv.relative_pose_ransac(np.array(a), np.array(b), "NISTER", 0.004, 1000)
    print M[:, 0:3]
    print M[:, 3]
    #M = pyopengv.relative_pose_fivept_nister(a, b)
    #print M

if __name__ == '__main__':
    path = sys.argv[1]
    
    config = Config.Config(path)
    data = Data.Data(path + '/images') 
    frame1, frame2 = Init(config, data)

    print "%s %s are good initial frame"%(frame1, frame2)
    Run(config, data, frame1, frame2)
