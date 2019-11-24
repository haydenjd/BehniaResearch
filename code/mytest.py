import scipy.ndimage.morphology as m
import numpy as np
import cv2
from pylab import *
import matplotlib.pyplot as plt


def skeletonize(img):
    h1 = np.array([[0, 0, 0],[0, 1, 0],[1, 1, 1]])
    m1 = np.array([[1, 1, 1],[0, 0, 0],[0, 0, 0]])
    h2 = np.array([[0, 0, 0],[1, 1, 0],[0, 1, 0]])
    m2 = np.array([[0, 1, 1],[0, 0, 1],[0, 0, 0]])
    hit_list = []
    miss_list = []
    for k in range(4):
        hit_list.append(np.rot90(h1, k))
        hit_list.append(np.rot90(h2, k))
        miss_list.append(np.rot90(m1, k))
        miss_list.append(np.rot90(m2, k))
    img = img.copy()
    while True:
        last = img
        for hit, miss in zip(hit_list, miss_list):
            hm = m.binary_hit_or_miss(img, hit, miss)
            img = np.logical_and(img, np.logical_not(hm))
        if np.all(img == last):
            break
    return img

img = cv2.imread("file3.jpg",0)
ret,img = cv2.threshold(img,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
img = 255 - img
img = cv2.dilate(img, element, iterations=3)

skel = skeletonize(img)
plt.imshow(skel, cmap="gray", interpolation="nearest")
plt.show()
