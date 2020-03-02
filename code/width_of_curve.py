import scipy.ndimage.morphology as m
import cv2
import numpy as np
from skimage import img_as_float
from skimage import io, color, morphology
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\scaaj\OneDrive\Documents\GitHub\BehniaResearch\pictures\file.jpg")

def thinning(name):
    img = image;
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    cv2.imshow("skel", skel)
    cv2.waitKey(0)


def thinning2(name):
    image = img_as_float(color.rgb2gray(io.imread(name)))
    image_binary = image < 0.5
    out_skeletonize = morphology.skeletonize(image_binary)
    out_thin = morphology.thin(image_binary)

    plt.imshow(out_skeletonize, cmap='gray')
    plt.show()


def binary(binaryFile):
    img = cv2.imread(binaryFile, 0)
    img = cv2.medianBlur(img, 5)
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite("binary.jpg", th3)
    plt.imshow(th3, 'gray')
    plt.show()
    if input_img is not None:
        clone = input_img.copy()
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, roi_selection)

        while True:
            cv2.imshow(window_name, input_img)

            if len(roi) == 4:
                input_img = clone.copy()
                roi = [0 if i < 0 else i for i in roi]
                cv2.rectangle(input_img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
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

                crop_img = clone[y1: y2, x1: x2]
                if len(crop_img):
                    cv2.namedWindow(window_crop_name, cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(window_crop_name, crop_img)

            k = cv2.waitKey(wait_time)
            if k == esc_keycode:
                cv2.imwrite("file1.jpg", crop_img)
                enlarge("file1.jpg")
                # erode("file3.jpg")
                binary("file3.jpg")
                erode("binary.jpg")
                thinning2("binary.jpg")

                s = skeletonize2("binary.jpg")
                plt.imshow(s)

                ###
                img = cv2.imread("erosion.jpg", 0)
                ret, img = cv2.threshold(img, 127, 255, 0)
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                img = 255 - img
                img = cv2.dilate(img, element, iterations=3)

                skel = skeletonize(img)
                # cv2.imwrite("Skeletonized.jpg", skel)
                # plt.imshow(skel, cmap="gray", interpolation="nearest")
                # plt.show()
                ###

                cv2.destroyAllWindows()
                break