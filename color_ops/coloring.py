import cv2
import numpy as np
import imutils
import math
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive

def shadow_removal(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # convert BGR to YUV
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) # perform histogram equalization of y channel
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # convert back to BGR
    # cv2.imshow('Color input image', image)
    # cv2.waitKey(0)
    # cv2.imshow('Histogram equalized', img_output)
    # cv2.imshow('Histogram equalized', img_yuv)
    # cv2.waitKey(0)
    return img_output

def gamma_correction(image, correction):
    image = image/255.0
    image = cv2.pow(image, correction)
    return np.uint8(image*255)

def colorOps(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv2.bilateralFilter
    # blurred = cv2.copyMakeBorder(blurred, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=[0, 0, 0])		 # added this 11/2/17. Trying to work with document occlusion. 
    edged = cv2.Canny(blurred, threshold1 = 75, threshold2 = 200) # was 0, 50...not sure what these numbers mean
    # edged_invert = edged.copy()
    # cv2.bitwise_not(edged, edged_invert)
    # plot_images([edged, edged_invert])
    return edged

def closed_inversion(image):
    closedEdges = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel = np.ones((5, 11))) # this potentially helps close Canny edges that are close but not quite touching
    closedInvert = cv2.bitwise_not(src = closedEdges.copy())
    
    kernel = np.ones((3,3),np.uint8) # original was 15x15. 5x5 is working well right now. 
    erosion = cv2.erode(closedInvert, kernel, iterations=1) # THIS LOOKS GOOD ON ALL but s4!! It erodes too much though. Losing some of document.
    # erosion = cv2.erode(closedEdges, kernel, iterations=1) # Attempting to return a non-inverted version for use with HoughLinesP in new.py -> DOESN'T WORK
    dilation = cv2.dilate(erosion,kernel,iterations = 1)

    reInvert = cv2.bitwise_not(src = erosion) # attempting to reInvert the eroded 

    # plot_images([closedEdges, closedInvert, erosion, dilation, reInvert]) # for testing 
    # return closedInvert
    return erosion # THIS WORKS WELL FOR PRETTY MUCH ALL DOCUMENTS SO FAR (except sample4)
    # return reInvert # return this for standard hough I guess, since I think it looks for white

def thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binarized = threshold_adaptive(gray, 251, offset = 10) # this is the original but it uses skimage
    # final = binarized.astype("uint8") * 255 # this is the original from skimage
    # gauss_thresh = cv2.adaptiveThreshold(src = gray, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #     thresholdType = cv2.THRESH_BINARY, blockSize = 251, C = 10) # blocksize was 11. This is good for a page
    gauss_thresh = cv2.adaptiveThreshold(src = gray, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 10) # blocksize was 11. This is good for a page
    return gauss_thresh


def dilation_canny(image):
    kernel = np.ones((5,5),np.uint8) # original was 15x15. 5x5 is working well right now. 
    dilation = cv2.dilate(image,kernel,iterations = 1)
    resized_image = standard_resize(image)
    resized_dilation = standard_resize(dilation)
    edged_resized = colorOps(resized_image)
    edged_dilation = colorOps(resized_dilation)
    plot_images([dilation, edged_resized, edged_dilation])