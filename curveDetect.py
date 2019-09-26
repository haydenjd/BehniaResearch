#!/usr/bin/env python

import numpy as np
import cv2

inputImage = cv2.resize((cv2.imread('crack.jpg')), (800,800))

#inputImage = cv2.imread('line2.jpg')

inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(inputImageGray,100,200,apertureSize = 3)

#edges = edgesOriginal[150:740, 80:760]

minLineLength = 500
maxLineGap = 0

lines = cv2.HoughLinesP(edges,1,np.pi/180,90,minLineLength,maxLineGap)

for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2)

#cv2.putText(inputImage,"Cracks detected", (500,250), font, 0.5, 255)

cv2.imshow("Result", inputImage)

cv2.waitKey(0)
cv2.destroyAllWindows()

