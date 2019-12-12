#!/usr/bin/env python

import cv2
import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
from skimage import morphology
#import numpy as np
import skimage

#Function to calculate the length
def getLength(img):
    image = cv2.imread(img)   
    height, width, channels = image.shape 
    print("Height: %d \nWidth: %d" % (height,width))
    endpoints = [[]]

    for y in range(0,height):
        for x in range(0,width):
            color = image[y,x]
            b = color[0]
            g = color[1]
            r = color[2]
            if b>150 and g>150 and r>150:
                if y>0 and x>0 and y<height-1 and x<width-1:
                    info = getColor(img,x,y)
                    count = info[[len(info)-1][0]]
                    if count>2:
                        image[y,x]=[255,0,0]
                    if count==2:
                        image[y,x]=[0,255,0]
                    if count==1:
                        image[y,x]=[0,0,255]
                        endpoints.append([x,y])

    endpoints.remove([])
    print(endpoints)
    l = 0.0
    for a in range(0,len(endpoints)):
        tf = True
        x = endpoints[a][0]
        y = endpoints[a][1]
        endpoints.remove([x,y])
        while(tf):
            image[y,x]=[255,255,0]
            info = getColor(img,x,y)
            count = info[[len(info)-1][0]]
            info.remove([])
            info.remove(count)
            check = 0
            for a1 in range(0,len(info)):
                color = image[info[a1-check][0],info[a1-check][1]]
                b = color[0]
                g = color[1]
                r = color[2]
                if b==255 and g==255 and r==0:
                    x=(info[a1-check][0])
                    y=(info[a1-check][1])
                    info.remove([x,y])
                    check = check + 1
            if len(info)==0:
                print("TOTAL LENGTH: %d pixels" % l)
                tf = False
            elif len(info)==1:
                x1 = x
                y1 = y
                x = info[0][1]
                y = info[0][0]
                if x==x1 or y==y1:
                    l = l+1.0
                else:
                    l = l+1.4142
            else:
                print("Splitting Point")
                break
        break
    

    cv2.imwrite('final.jpg',image)
    cv2.imshow("final",image)
    cv2.waitKey(0)



def getColor(img,x,y):
    image = cv2.imread(img)
    height, width, channels = image.shape
    info = [[]]
    count = 0
    color = image[y,x]
    b = color[0]
    g = color[1]
    r = color[2]
    color1 = image[y-1,x-1]
    b1 = color1[0]
    g1 = color1[1]
    r1 = color1[2]
    if b1>150 and g1>150 and r1>150:
        count = count+1
        info.append([y-1,x-1])
    color2 = image[y-1,x]
    b2 = color2[0]
    g2 = color2[1]
    r2 = color2[2]
    if b2>150 and g2>150 and r2>150:
        count = count+1
        info.append([y-1,x])
    color3 = image[y-1,x+1]
    b3 = color3[0]
    g3 = color3[1]
    r3 = color3[2]
    if b3>150 and g3>150 and r3>150:
        count = count+1
        info.append([y-1,x+1])
    color4 = image[y,x-1]
    b4 = color4[0]
    g4 = color4[1]
    r4 = color4[2]
    if b4>150 and g4>150 and r4>150:
        count = count+1
        info.append([y,x-1])
    color5 = image[y,x+1]
    b5 = color5[0]
    g5 = color5[1]
    r5 = color5[2]
    if b5>150 and g5>150 and r5>150:
        count = count+1
        info.append([y,x+1])
    color6 = image[y+1,x-1]
    b6 = color6[0]
    g6 = color6[1]
    r6 = color6[2]
    if b6>150 and g6>150 and r6>150:
        count = count+1
        info.append([y+1,x-1])
    color7 = image[y+1,x]
    b7 = color7[0]
    g7 = color7[1]
    r7 = color7[2]
    if b7>150 and g7>150 and r7>150:
        count = count+1
        info.append([y+1,x])
    color8 = image[y+1,x+1]
    b8 = color8[0]
    g8 = color8[1]
    r8 = color8[2]
    if b8>150 and g8>150 and r8>150:
        count = count+1
        info.append([y+1,x+1])

    info.append(count)
    return info

################################################################


## read the image, grayscale it, binarize it, then remove small pixel clusters
im = cv2.imread('gaps.jpg')
#grayscale = skimage.color.rgb2gray(im)
#binarized = np.where(grayscale>0.1, 1, 0)
#processed = morphology.remove_small_objects(binarized.astype(bool), min_size=2, connectivity=2).astype(int)
#
## black out pixels
#mask_x, mask_y = np.where(processed == 0)
#im[mask_x, mask_y, :3] = 0
#
#cv2.imshow('Spots',im)
#cv2.imwrite('spots.jpg',im)
#
## plot the result
##plt.figure(figsize=(10,10))
##plt.imshow(im)
#

kernel = np.ones((5,5),np.uint8)
#opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('close.jpg',closing)

#getLength('gaps_filled.jpg')
getLength('close.jpg')

#getColor('gaps.jpg',265,151)
