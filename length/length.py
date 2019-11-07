#!/usr/bin/env python

import numpy as np
import cv2

def nothing(x):
    pass

################################################

def image(name):
    #Reads the jpg image of the cracked material and changes the size to fit in the screen
    orig = cv2.resize((cv2.imread(name)), (800,800))
    
    #Reads the jpg image of the cracked material and makes the picture black and white
    img = cv2.imread(name, 0)
    
    #Resizes the image to fit in the screen
    #img2 = cv2.resize(img, (800,800))
    img2 = img

    cv2.namedWindow('cropped')
    cv2.createTrackbar('X','cropped',0,500,nothing)
    cv2.createTrackbar('Y','cropped',0,500,nothing)
    cv2.createTrackbar('Is the image set?','cropped',0,1,nothing)
    
    edges = cv2.Canny(img2, 100, 200)
    
    #Resizes the image to the demensions that only include the image
    #cropped = edges[150:740, 80:760]
    cropped = img2
    
    
    while(1):
        cv2.imshow('cropped',cropped)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        
        x = cv2.getTrackbarPos('X','cropped')
        y = cv2.getTrackbarPos('Y','cropped')
        done = cv2.getTrackbarPos('Is the image set?','cropped')
        #Uses canny to only detect the edges
        #edeges = cv2.Canny(img2, 100, 300)
        edges = cv2.Canny(img2, x, y)
    
        #Resizes the image to the demensions that only include the image
        cropped = edges
    
        if done == 1:
            break

    cv2.imwrite("file.jpg", cropped)

###############################################
def pixels(name):
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
    #print "Height: " + str(height)
    #print "width: " + str(width)
    #print "Crack pixels: " + str(count)
    #print "File of all crack coordinates saved in rgb"

###############################################

def display():
    #Displays the original and final picture
    cv2.imshow('Original', orig)
    #cv2.imshow('Cropped', cropped)
    
    #Saves the final picture to the current directory
    #cv2.imwrite("file.jpg", cropped)
    cv2.imwrite("cracking.jpg", orig)
    
    #Gets rid of the pictures when you exit out of them
    cv2.waitKey(0)
    cv2.destroyAllWindows()

###############################################
###############################################
###############################################
###############################################
#Main
filename = 'cracking.jpg'
#image(filename)

#Cropping the image
selection = False
roi = []

def roi_selection(event, x, y, flags, param):
	global selection, roi
	if event == cv2.EVENT_LBUTTONDOWN:
		selection = True
		roi = [x, y, x, y]
	elif event == cv2.EVENT_MOUSEMOVE:
		if selection == True:
			roi[2] = x
			roi[3] = y

	elif event == cv2.EVENT_LBUTTONUP:
		selection = False
		roi[2] = x
		roi[3] = y

image_read_path=filename
window_name='Input Image'
window_crop_name='Cropped Image'
esc_keycode=27
wait_time=1
input_img = cv2.imread(image_read_path,cv2.IMREAD_UNCHANGED)

if input_img is not None:
	clone = input_img.copy()
	cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)
	cv2.setMouseCallback(window_name, roi_selection)

	while True:
		cv2.imshow(window_name,input_img)

		if len(roi) == 4:
			input_img = clone.copy()
			roi = [0 if i < 0 else i for i in roi]
			cv2.rectangle(input_img, (roi[0],roi[1]), (roi[2],roi[3]), (0, 255, 0), 2)
			if roi[0] > roi[2]:
				x1 = roi[2]
				x2 = roi[0]
			else:
				x1 = roi[0]
				x2 = roi[2]
			if roi[1] > roi[3]:
				y1 = roi[3]
				y2 = roi[1]
			else:
				y1 = roi[1]
				y2 = roi[3]

			crop_img = clone[y1 : y2 , x1 : x2]
			if len(crop_img):
				cv2.namedWindow(window_crop_name,cv2.WINDOW_AUTOSIZE)
				cv2.imshow(window_crop_name,crop_img)

		k = cv2.waitKey(wait_time)
		if k == esc_keycode:
                    cv2.imwrite("file.jpg", crop_img)
                    image("file.jpg")
                    #pixels("file.jpg")
                    #display()
                    cv2.destroyAllWindows()
                    break

else:
	print("Please Check The Path of Input File")

