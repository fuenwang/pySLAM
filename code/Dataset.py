import os
import sys
import cv2
import numpy as np
import scipy.io as io

class Dataset:
    def __init__(self, path):
        self.root = path
        self.lst = [x for x in sorted(os.listdir(path)) if x.endswith('.mat')]
    
    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        path = '%s/%s'%(self.root, self.lst[idx])
        
        mat = io.loadmat(path)['data'][0][0]
        img = cv2.cvtColor(mat[1], cv2.COLOR_RGB2BGR)
        depth = mat[0]

        return [img, depth]

if __name__ == '__main__':
    data = Dataset('../data')

    [img, depth] = data[20]
    mx = np.max(depth)
    #depth[depth == 0] = -1
    print depth
    depth /= mx
    #'''
    cv2.namedWindow('GG')
    #cv2.namedWindow('DD')
    cv2.imshow('GG', depth)
    #cv2.imshow('DD', img)
    cv2.waitKey(0)
    #'''
