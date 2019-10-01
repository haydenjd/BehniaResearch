#!/usr/bin/env python

import numpy as np
import cv2

################################################
#Reads the jpg image of the cracked material and changes the size to fit in the screen
orig = cv2.resize((cv2.imread('crack.jpg')), (800,800))

#Reads the jpg image of the cracked material and makes the picture black and white
img = cv2.imread('crack.jpg', 0)

#Resizes the image to fit in the screen
img2 = cv2.resize(img, (800,800))

#Uses canny to only detect the edges
#edges = cv2.Canny(img2, 100, 300)
edges = cv2.Canny(img2, 100, 10) 

#Resizes the image to the demensions that only include the image
cropped = edges[150:740, 80:760]

###############################################

#Gets the height and width of the cropped picture
height = np.size(cropped, 0)
width = np.size(cropped, 1)
count = 0

f = open("rgb","w+")

#Using a nested for loop to go through all pixels, prints and counts all the white ones
for x in range(0,height):
    for y in range(0,width):
        color = int(edges[x,y])
        if color == 255:
            f.write("[%d,%d] " % (x , y))
            count = count + 1

#Prints out the height, width and number of pixels that are part of a crack
print "Height: " + str(height)
print "width: " + str(width)
print "Crack pixels: " + str(count)
print "File of all crack coordinates saved in rgb"

###############################################
#Displays the original and final picture
cv2.imshow('Original', orig)
cv2.imshow('Cropped', cropped)

#Saves the final picture to the current directory
cv2.imwrite("file.jpg", cropped)

#Gets rid of the pictures when you exit out of them
cv2.waitKey(0)
cv2.destroyAllWindows()

