#!/usr/bin/env python

import numpy as np
import cv2

name = 'cracking.jpg'

selection = False
roi = []

def nothing(x):
    pass

def canny(filename):
    img2 = cv2.imread('file.jpg',0)
    cv2.namedWindow('cropped')
    cv2.createTrackbar('X','cropped',0,500,nothing)
    cv2.createTrackbar('Y','cropped',0,500,nothing)
    cv2.createTrackbar('Is the image set?','cropped',0,1,nothing)

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
        cropped = edges[150:740, 80:760]

        if done == 1:
            break

    cv2.imwrite("file.jpg", cropped)


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
			
image_read_path=name
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
                    canny("file.jpg")
                    cv2.destroyAllWindows()
		    break
			
else:
	print 'Please Check The Path of Input File'

