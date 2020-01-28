#Python Code that will take in a picture that contains cracking
#Outputs the length of the cracking

import math
import scipy.ndimage.morphology as m
import cv2
import numpy as np
from skimage import img_as_float
from skimage import io, color, morphology
from skimage import io, morphology, img_as_bool, segmentation
from scipy import ndimage as ndi
import matplotlib.pyplot as plt


def thinning2(name):
    image = img_as_float(color.rgb2gray(io.imread(name)))
    image_binary = image < 0.5
    out_skeletonize = morphology.skeletonize(image_binary)
    out_thin = morphology.thin(image_binary)
    
    plt.imsave('gaps.jpg', out_skeletonize, cmap='gray')
    plt.imsave('thin.jpg', out_thin, cmap='gray')
    img = cv2.imread("gaps.jpg")
    img2 = cv2.imread("thin.jpg")
    cv2.imshow("Thinning2", img2)
    cv2.waitKey(0)


def binary(binaryFile):
    img = cv2.imread(binaryFile,0)
    img = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    cv2.imwrite("binary.jpg", th3)
    cv2.imshow("Binary",th3)
    cv2.waitKey(0)

def enlarge(fileName):
    img = cv2.imread(fileName)
    cv2.imwrite("file3.jpg", cv2.resize(img, (a,b)))

def dilate(fileName):
    img = cv2.imread(fileName)
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    #cv2.imshow("dilation",dilation)

def erode(fileName):
    img = cv2.imread(fileName,0)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    #cv2.imshow("erosion",erosion)
    cv2.imwrite("erosion.jpg",erosion)

def median(img1):
    img = cv2.imread(img1)
    median = cv2.medianBlur(img, 3)
    #compare = np.concatenate((img, median), axis=1) #side by side comparison
    cv2.imwrite("median.jpg",median)
    #cv2.imshow('img', median)
    cv2.waitKey(0)

def noiseReduce(img1):
    img = cv2.imread(img1)
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    cv2.imshow('Noise Reduction', dst)
    cv2.waitKey(0)

def complete(img):
    image = img_as_bool(io.imread(img))
    out = ndi.distance_transform_edt(~image)
    out = out < 0.02 * out.max()
    out = morphology.skeletonize(out)
    #out = morphology.binary_dilation(out, morphology.selem.disk(1))
    out = segmentation.clear_border(out)
    out = out | image
    
    cv2.imshow("out",out)
    cv2.waitKey(0)
    
    cv2.imwrite('gaps_filled.jpg', out)
    #plt.imsave('gaps_filled.jpg', out, cmap='gray')

def opening(img1):
    img = cv2.imread(img1)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("Opening",opening)
    cv2.waitKey(0)
    cv2.imwrite("opening.jpg",opening)

def closing(img1):
    img = cv2.imread(img1)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("Closing",closing)
    cv2.waitKey(0)
    cv2.imwrite("closing.jpg",closing)

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
    #cv2.imshow("bandw", image)
    cv2.imwrite("bandw.jpg", image)

#Function to calculate the length
def getLength(img,wU,hU,units):
    image = cv2.imread(img)
    height, width, channels = image.shape
    w = wU/width
    h = hU/height
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
                if y==0 or x==0 or y==height or x==width:
                    image[y,x]=[0,0,255]
                    endpoints.append([x,y])

    endpoints.remove([])
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
            x1=(info[0][0])
            y1=(info[0][1])
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
                #info.remove!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if len(info)==0:
                print("TOTAL LENGTH: %.3f %s" % (l,units))
                tf = False
            elif len(info)==1:
                x1 = x
                y1 = y
                x = info[0][1]
                y = info[0][0]
                if x==x1 or y==y1:
                    l = l+w
                elif y==y1:
                    l = l+h
                else:
                    l = l+math.sqrt(math.pow(w,2)+math.pow(h,2))
            else:
                #info.add([x,y])
                print("Splitting Point")
                print("TOTAL LENGTH: %d pixels" % l)
                #break
        break


    cv2.imwrite('end.jpg',image)
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

#############################################MAIN###################################
img = cv2.imread('split2.PNG')
height, width, channels = img.shape
#a = int(width/5) 
a = width
#b = int(height/5) 
b = height
widthUnits = 9.0
heightUnits = 6.0
units = "inches"
cv2.imwrite("file2.jpg", cv2.resize(img, (a,b)))
filename = 'file2.jpg'
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
                    cv2.imwrite("file1.jpg", crop_img)
                    enlarge("file1.jpg")
                    binary("file3.jpg")
                    erode("binary.jpg")
                    median("binary.jpg")
                    thinning2("median.jpg")
                    #noiseReduce("gaps.jpg")
                    #closing("gaps.jpg")
                    complete("gaps.jpg")
                    bandw("gaps_filled.jpg")
                    getLength("bandw.jpg",widthUnits, heightUnits, units)
                    cv2.destroyAllWindows()
                    break

else:
        print("Please Check The Path of Input File")

