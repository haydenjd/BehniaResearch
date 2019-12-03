#Test File

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
    img = cv2.imread("gaps.jpg")
    cv2.imshow("Thinning", img)
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
    cv2.imwrite("file3.jpg", cv2.resize(img, (800,800)))

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


def complete(img):
    image = img_as_bool(io.imread(img))
    out = ndi.distance_transform_edt(~image)
    out = out < 0.05 * out.max()
    out = morphology.skeletonize(out)
    #out = morphology.binary_dilation(out, morphology.selem.disk(1))
    out = segmentation.clear_border(out)
    out = out | image
    
    cv2.imshow("out",out)
    cv2.waitKey(0)
    
    plt.imsave('gaps_filled.jpg', out, cmap='gray')


#############################################MAIN###################################
img = cv2.imread('crack1.jpg')
cv2.imwrite("file2.jpg", cv2.resize(img, (800,800)))
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
                    thinning2("binary.jpg")
                    complete("gaps.jpg")
                    cv2.destroyAllWindows()
                    break

else:
        print("Please Check The Path of Input File")

