#Python Code that will take in a picture that contains cracking
#Outputs the length of the cracking

#All the imports for packages used
import shutil
import os
import math
import scipy.ndimage.morphology as m
import cv2
import numpy as np
from skimage import img_as_float
from skimage import io, color, morphology
from skimage import io, morphology, img_as_bool, segmentation
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import ctypes
import tkinter as tk
import tkinter.filedialog
from tkinter.filedialog import askopenfilename

#Makes each pixel of the image black or white
def binary(img):
    im_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("binary.jpg", im_bw)

#Applies median filtering to get rid of noise
def median(img1):
    img = cv2.imread(img1)
    median = cv2.medianBlur(img, 3)
    cv2.imwrite("median.jpg",median)

#Thins the image to one pixel wide using the skeletonize function from morphology
def thinning2(name):
    image = img_as_float(color.rgb2gray(io.imread(name)))
    image_binary = image < 0.5
    out_skeletonize = morphology.skeletonize(image_binary)
    out_thin = morphology.thin(image_binary)
    plt.imsave('gaps.jpg', out_skeletonize, cmap='gray')

#Finds the edges using the Canny Edge Detection
def canny(name):
    image = cv2.imread(name)
    edges = cv2.Canny(image, 100, 200)
    cv2.imwrite("canny.jpg",edges)

#Fills in gaps in the skeleton image
def complete(img):
    image = img_as_bool(io.imread(img))
    out = ndi.distance_transform_edt(~image)
    out = out < 0.02 * out.max()
    out = morphology.skeletonize(out)
    out = segmentation.clear_border(out)
    out = out | image
    cv2.imwrite('gaps_filled.jpg', out)

#Makes all green pixels white
def bandw(img1):
    image = cv2.imread(img1)
    height, width, channels = image.shape
    for y in range(0,height):
        for x in range(0,width):
            color = image[y,x]
            b = color[0]
            g = color[1]
            r = color[2]
            if b>0 and g>100 and r>0:
                image[y,x] = [255,255,255]
            else:
                image[y,x] = [0,0,0]
    cv2.imwrite("bandw.jpg", image)

#The switch function is used to assign each number of crack with a color
def switch(x):
    return {
        1: 'Red',
        2: 'Yellow',
        3: 'White',
        4: 'Purple',
        5: 'Orange',
        6: 'Pink'
    }.get(x, 'Color')

#Function to calculate the length
def getLength(img,wU,hU,units,filename):
    image = cv2.imread(img)
    height, width, channels = image.shape
    w = wU/width
    h = hU/height
    #Array created that will contain all endpoints of the cracks
    endpoints = [[]]
    split = [[]]

    for y in range(0,height):
        for x in range(0,width):
            color = image[y,x]
            #RGB values gotten for the selected pixel
            b = color[0]
            g = color[1]
            r = color[2]
            #Checks to see if all RGBs are part of the crack
            if b>150 and g>150 and r>150:
                #Cracks to see if pixel is not part of the border
                if y>0 and x>0 and y<height-1 and x<width-1:
                    #The getColor function takes in a selected pixel and checks the eight pixels around it to see if it is part of the cracking
                    info = getColor(img,x,y)
                    #Count is the number of pixels around the selected one that are part of the cracking
                    count = info[[len(info)-1][0]]
                    #If there are more than 2 pixels in count, the selected pixel is a splitting point
                    if count>2:
                        image[y,x]=[255,0,0]
                        split.append([x,y])
                        #print("[%d,%d]" % (x,y))
                    #If there are exactly two pixels in count, the selected pixel is in the middle of the cracking
                    if count==2:
                        image[y,x]=[0,255,0]
                    #If there is exactly one pixel in count, the selected pixel is an endpoint and added to the endpoint array
                    if count==1:
                        image[y,x]=[0,0,255]
                        endpoints.append([x,y])
                #Categorizes the pixel as an endpoint if the selected pixel is on the border of the image
                if y==0 or x==0 or y==height or x==width:
                    image[y,x]=[0,0,255]
                    endpoints.append([x,y])

    endpoints.remove([])
    intersection = [[]]
    slopes = []
    #l is the variable to keep track of the length
    l = 0.0
    name = filename + "_output.txt"
    f= open(name,"w+")
    #Loops through all the endpoints
    #For each endpoint we start with that pixel and move throughout the crack, adding to the length for each pixel until we get to another endpoint
    #counter is used to asign a number to each crack
    counter = 1
    while(len(endpoints)>0):
        tf = True
        x = endpoints[0][1]
        y = endpoints[0][0]
        endpoints.remove([y,x])
        while(tf):
            #Colors each crack based on the counter number
            if counter == 1:
                image[x,y]=[0,0,255]
            elif counter == 2:
                image[x,y]=[0,255,255]
            elif counter == 3:
                image[x,y]=[255,255,255]
            elif counter == 4:
                image[x,y]=[255,0,255]
            elif counter == 5:
                image[x,y]=[0,127,255]
            elif counter == 6:
                image[x,y]=[127,0,255]
            else:
                image[x,y]=[127,127,255]
            #Calls the getColor function to see which pixels are next to the selected one
            info = getColor(img,y,x)
            #Gets a number for how many pixels are next the selected one
            count = info[[len(info)-1][0]]
            info.remove([])
            info.remove(count)
            if count == 2:
                s = getSlope(img,y,x)
                slopes.append([s,x,y])
            check = 0
            #Loops through all the endpoints and splitting points
            for a1 in range(0,len(info)):
                color = image[info[a1-check][0],info[a1-check][1]]
                b = color[0]
                g = color[1]
                r = color[2]
                if r==255:
                    x=(info[a1-check][0])
                    y=(info[a1-check][1])
                    info.remove([x,y])
                    check = check + 1
            if len(info) == 0 or [y,x] in endpoints:
                if [y,x] in endpoints:
                    endpoints.remove([y,x])
                tf = False
            elif len(info) == 1:
                x1 = x
                y1 = y
                x = info[0][0]
                y = info[0][1]
                if x==x1:
                    l = l+w
                elif y==y1:
                    l = l+h
                else:
                    l = l+math.sqrt(math.pow(w,2)+math.pow(h,2))
            else:
                intersection.append([x,y])
                for i in range(0,len(info)):
                    endpoints.append([info[i][1],info[i][0]])
                image[x,y]=[0,0,0]
                image[x,y]=[255,255,255]
                tf = False
        if l != 0:
            f.write("Length %d (%s): %.3f %s\n" % (counter,switch(counter),l,units))
            counter = counter + 1
        l = 0
    f.close()
    cv2.imwrite('end.jpg',image)
    return slopes

#Takes in the skeleton image and finds the perpendicular slope for each part of the cracking
#Slope using 2-4 other pixels to get a more accurate length
def getSlope(img,x,y):
    total = 0
    counter = 0
    slope = 0
    info = getColor(img,x,y)
    #We look to get at least the two points around the selected one, but will hopefully get up to four around it
    point1 = [0,0]
    point2 = [info[1][1], info[1][0]]
    point3 = [x,y]
    point4 = [info[2][1], info[2][0]]
    point5 = [0,0]
    infoP2 = getColor(img,point2[0],point2[1])
    infoP4 = getColor(img,point4[0],point4[1])
    if infoP2[len(infoP2)-1] == 2:
        infoP2.remove(2)
        infoP2.remove([y,x])
        infoP2.remove([])
        point1 = [infoP2[0][1],infoP2[0][0]]
        counter = counter + 1
    if infoP4[len(infoP4)-1] == 2:
        infoP4.remove(2)
        infoP4.remove([y,x])
        infoP4.remove([])
        point5 = [infoP4[0][1],infoP4[0][0]]
        counter = counter + 5
    if ((float(point2[0]-point1[0])) != 0) and (counter == 1 or counter == 6):
        slope = slope + (float(point2[1]-point1[1])/(float(point2[0]-point1[0])))
        total = total + 1
    if (float(point3[0]-point2[0])) != 0:
        slope = slope + (float(point3[1]-point2[1])/(float(point3[0]-point2[0])))
        total = total + 1
    if (float(point4[0]-point3[0])) != 0:
        slope = slope + (float(point4[1]-point3[1])/(float(point4[0]-point3[0])))
        total = total + 1
    if ((float(point5[0]-point4[0])) != 0) and (counter == 5 or counter == 6):
        slope = slope + (float(point5[1]-point4[1])/(float(point5[0]-point4[0])))
        total = total + 1

    if total == 0:
        slope = 1
    else:
        slope = slope/total
    return slope

#Takes in the canny image and the slopes and returns the widths
def getWidth(canny1,slopes,wU,hU,units):
    canny = cv2.imread(canny1)
    #WIdth and height in pixels and in the propper units are now variables
    height, width, channels = canny.shape
    w = wU/width
    h = hU/height
    #Variables for high, low, avg and an array of all widths are initialized and will be changed to what they should be throughout the code
    total = []
    high = 0
    low = 100
    avg = 0
    #Loops through the array of all the slopes of the pixels
    for a in range(0,len(slopes)):
        slope = slopes[a][0]
        x = slopes[a][1]
        y = slopes[a][2]
        canny[x,y]=[0,0,255]
        #Slope is put into fraction form to get the x and y of the slope
        newSlope = (float(slope)).as_integer_ratio()
        slopeX = newSlope[1]
        slopeY = newSlope[0]
        check = True
        counter = 1
        #Slope is used to extend the length until it reaches the edges of the cracking
        while check:
            #Checks to see if slope causes the width to go out of bounds
            if y-(counter*slopeY) >= width or y+(counter*slopeY) >= width or y-(counter*slopeY) < 0 or y+(counter*slopeY) < 0 or x-(counter*slopeX) >= width or x+(counter*slopeX) >= width or x-(counter*slopeX) < 0 or x+(counter*slopeX) < 0:
                #print("Out of bounds")
                break
            #Slope is used to add to the width
            color1 = canny[x+(counter*slopeX),y-(counter*slopeY)]
            color2 = canny[x-(counter*slopeX),y+(counter*slopeY)]
            if (color1[0]>200 and color1[1]>200 and color1[2]) or (color2[0]>200 and color2[1]>200 and color2[2]):
                check = False
                total.append(math.sqrt(2*(math.pow((counter*w*slopeX),2)+math.pow((counter*h*slopeY),2))))
                canny = cv2.line(canny, (y+(counter*slopeY), x-(counter*slopeX)), (y-(counter*slopeY), x+(counter*slopeX)),(0,0,255))
            counter = counter + 1
    #The average width is calculated
    for b in range(0,len(total)):
        if total[b] < low:
            low = total[b]
        if total[b] > high:
            high = total[b]
        avg = avg + total[b]
    avg = avg/len(total)
    #Avg, smallest and largest lengths printed out
    #Saves and shows the widths on the canny image
    cv2.imwrite('slopes.jpg',canny)

#Returns an array with the pixels that are part of the cracking from the eight surrounding pixels
#Returns the number of pixels that is part of the cracks to allow the code to categorize as either endpoint, splitting point or regular pixel
def getColor(img,x,y):
    #Reads in the image
    image = cv2.imread(img)
    #Gets the height and width values (in pixels)
    height, width, channels = image.shape
    info = [[]]
    count = 0
    #Gets the RGB values for the selected pixel
    color = image[y,x]
    b = color[0]
    g = color[1]
    r = color[2]
    #Checks to see if the pixel to the upper left is part of the cracking
    color1 = image[y-1,x-1]
    b1 = color1[0]
    g1 = color1[1]
    r1 = color1[2]
    if b1>150 and g1>150 and r1>150:
        count = count+1
        info.append([y-1,x-1])
    #Checks to see if the pixel above is part of the cracking
    color2 = image[y-1,x]
    b2 = color2[0]
    g2 = color2[1]
    r2 = color2[2]
    if b2>150 and g2>150 and r2>150:
        count = count+1
        info.append([y-1,x])
    #Checks to see if the pixel to the upper right is part of the cracking
    color3 = image[y-1,x+1]
    b3 = color3[0]
    g3 = color3[1]
    r3 = color3[2]
    if b3>150 and g3>150 and r3>150:
        count = count+1
        info.append([y-1,x+1])
    #Checks to see if the pixel to the left is part of the cracking
    color4 = image[y,x-1]
    b4 = color4[0]
    g4 = color4[1]
    r4 = color4[2]
    if b4>150 and g4>150 and r4>150:
        count = count+1
        info.append([y,x-1])
    #Checks to see if the pixel to the right is part of the cracking
    color5 = image[y,x+1]
    b5 = color5[0]
    g5 = color5[1]
    r5 = color5[2]
    if b5>150 and g5>150 and r5>150:
        count = count+1
        info.append([y,x+1])
    #Checks to see if the pixel to the bottom left is part of the cracking
    color6 = image[y+1,x-1]
    b6 = color6[0]
    g6 = color6[1]
    r6 = color6[2]
    if b6>150 and g6>150 and r6>150:
        count = count+1
        info.append([y+1,x-1])
    #Checks to see if the pixel below is part of the cracking
    color7 = image[y+1,x]
    b7 = color7[0]
    g7 = color7[1]
    r7 = color7[2]
    if b7>150 and g7>150 and r7>150:
        count = count+1
        info.append([y+1,x])
    #Checks to see if the pixel to the bottom right is part of the cracking
    color8 = image[y+1,x+1]
    b8 = color8[0]
    g8 = color8[1]
    r8 = color8[2]
    if b8>150 and g8>150 and r8>150:
        count = count+1
        info.append([y+1,x+1])

    info.append(count)
    return info

#############################################MAIN###################################
#User finds file in file system

pathA = "C:\\Users\\jimmy\\Documents\\Behnia Research\\BehniaResearch\\images"
pathB = "C:\\Users\\jimmy\\Documents\\Behnia Research\\BehniaResearch"
files = [i for i in os.listdir(pathA) if i.endswith("PNG") or i.endswith("png") or i.endswith("jpg") or i.endswith("JPG")]
for file in files:
    print(file)
    path1 = pathA + "\\" + file
    path2 = pathB + "\\" + file
    #os.rename(path1, path2)
    shutil.copy(path1, path2)
    #os.replace(path1, path2)
    filename = file
    img = cv2.imread(filename)
    
    widthUnits = 9
    heightUnits = 6
    units = "inches"
    height, width, channels = img.shape
    
    #Image is resized
    cv2.imwrite("resize.jpg", cv2.resize(img, (int(width),int(height))))
    filename = 'resize.jpg'
    selection = False
    roi = []
    crop_img = [0]
    crop_img = img
    #Created file cropped.jpg that saves the newly cropped image
    cv2.imwrite("cropped.jpg", crop_img)
    #Image is transformed into a binary image
    binary("cropped.jpg")
    #Uses the Canny Edge Detection to find the edges of the cracking
    canny("binary.jpg")
    #Median Filtering is used to get rid of access points
    median("binary.jpg")
    #Thinning is used to make the cracking one pixel wide
    thinning2("median.jpg")
    #Complete fills in gaps in the cracking
    complete("gaps.jpg")
    #Cracking RGB values are 255
    bandw("gaps_filled.jpg")
    #Length is calculated using demensions and units given. Perpendicular slopes are returned
    slopes = getLength("bandw.jpg",widthUnits, heightUnits, units, file)
    #Width is calculated
    getWidth("canny.jpg", slopes, widthUnits, heightUnits, units)
    #Windows are removed
    cv2.destroyAllWindows()
