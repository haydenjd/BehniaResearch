#!/usr/bin/env python

import cv2
import numpy as np
from PIL import Image


#Function to calculate the length
def getLength(img):
    im = Image.open(img)   
    out = Image.new('I', im.size, 0xffffff)

    width, height = im.size
    for x in range(width):
        for y in range(height):
            [r,g,b] = im.getpixel((x,y))
            if([r,g,b]==[255,255,255]):
                print("YES")
    out.save('final.jpg')

################################################################

getLength('gaps_filled.jpg')
