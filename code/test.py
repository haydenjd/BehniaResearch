#Test File

import scipy.ndimage.morphology as m
import cv2
import numpy as np
from skimage import img_as_float
from skimage import io, color, morphology
import matplotlib.pyplot as plt

def thinning(name):
    img = cv2.imread(name,0)
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
     
    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
     
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
     
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
     
    cv2.imshow("skel",skel)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

def thinning2(name):
    image = img_as_float(color.rgb2gray(io.imread(name)))
    image_binary = image < 0.5
    out_skeletonize = morphology.skeletonize(image_binary)
    out_thin = morphology.thin(image_binary)
    
    plt.imshow(out_skeletonize, cmap='gray')
    plt.show()

def binary(binaryFile):
    img = cv2.imread(binaryFile,0)
    img = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    cv2.imwrite("binary.jpg", th3)
    plt.imshow(th3,'gray')
    plt.show()

def select(file1):
    image = cv2.imread('file1.jpg')
    blur = cv2.medianBlur(image, 7)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3)

    canny = cv2.Canny(thresh, 120, 255, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    dilate = cv2.dilate(opening, kernel, iterations=2)

    cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 3000
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:
            cv2.drawContours(image, [c], -1, (36, 255, 12), 2)

    cv2.imshow('image', image)
    cv2.imwrite('image.png', image)
    cv2.waitKey(0)

def enlarge(fileName):
    img = cv2.imread(fileName)
    cv2.imwrite("file3.jpg", cv2.resize(img, (800,800)))

def dilate(fileName):
    img = cv2.imread(fileName)
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    cv2.imshow("dilation",dilation)

def erode(fileName):
    img = cv2.imread(fileName,0)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    cv2.imshow("erosion",erosion)
    cv2.imwrite("erosion.jpg",erosion)

def skel(img):
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    #apply Canny edge detection
    canny_img = cv2.Canny(blurred, 150, 200)
    
    #apply Distance Transform
    invert_canny= 255- canny_img
    dist_trans= cv2.distanceTransform(invert_canny, cv2.DIST_L2, 3)
    
    #normalize to visualize dist-transformed img
    cv2.normalize(dist_trans, dist_trans, 0.0, 1.0, cv2.NORM_MINMAX)
    
    # apply dilation
    kernel=np.ones((5,5), np.uint8)
    dilate=cv2.dilate(dist_trans,kernel, iterations=1)

    cv2.imshow("dilate",dilate)
    cv2.waitKey(0)

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

def skeletonize2(img):

    #img = img.copy() # don't clobber original
    skel = cv2.imread(img) #= img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break
    
    return skel

#############################################MAIN###################################
img = cv2.imread('blackwhite.jpg')
cv2.imwrite("file2.jpg", cv2.resize(img, (800,800)))
filename = 'file2.jpg'
selection = False
roi = []
dist = cv2.distanceTransform(img,cv2.DIST_L2,3)
_,mv,_,mp = cv2.minMaxLoc(dist)
print(mv*2, mp)
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
                    #erode("file3.jpg")
                    binary("file3.jpg")
                    erode("binary.jpg")
                    thinning2("binary.jpg")
                    
                    s = skeletonize2("binary.jpg")
                    plt.imshow(s)

                    ###
                    img = cv2.imread("erosion.jpg",0)
                    ret,img = cv2.threshold(img,127,255,0)
                    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
                    img = 255 - img
                    img = cv2.dilate(img, element, iterations=3)
                    
                    skel = skeletonize(img)
                    #cv2.imwrite("Skeletonized.jpg", skel)
                    #plt.imshow(skel, cmap="gray", interpolation="nearest")
                    #plt.show()
                    ###

                    cv2.destroyAllWindows()
                    break

else:
        print("Please Check The Path of Input File")

