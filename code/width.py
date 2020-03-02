from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from skimage.morphology import skeletonize
from skimage.util import invert

image = cv2.imread(r"C:\Users\scaaj\OneDrive\Documents\GitHub\BehniaResearch\pictures\file1.jpg")
invertImage = invert(image)
(thresh, binaryImage) = cv2.threshold(invertImage, 128, 255, cv2.THRESH_BINARY)
skeleton = skeletonize(binaryImage)
cv2.imshow("inverted image",invertImage)
cv2.imshow("binary image",binaryImage)
cv2.imshow("skelton image",skeleton)
cv2.waitKey(0)