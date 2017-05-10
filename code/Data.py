import os
import sys
import cv2
import numpy as np

class Data:
    def __init__(self, path):
        self._root_path = path
        self._img_lst = ['%s/%s'%(path, x) for x in sorted(os.listdir(path))]
        self._index = 0

    def Next(self, grayscale = True, normalized = True):
        if self._index == len(self._img_lst):
            print "No other images"
            exit()
        #name = self._img_lst[self._index]
        short = self._img_lst[self._index].split('/')[-1]
        if grayscale:
            img = cv2.imread(self._img_lst[self._index], cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(self._img_lst[self._index], cv2.IMREAD_COLOR)
        self._index += 1

        if normalized:
            img = img.astype(np.float32) / 255
        
        return short, img

    def ReadImg(self, short, grayscale = True, normalized = True):
        path = '%s/%s'%(self._root_path, short)
        if grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)

        if normalized:
            img = img.astype(np.float32) / 255
        return img
