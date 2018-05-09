
import cv2
import sys
import numpy as np
import argparse
import imutils
import math
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive
import functions
import utility
import demo
import hough


# <---- COMPARISON METHODS -----> #

def blur_comparison(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = image.copy()
    
    bilateral = cv2.bilateralFilter(gray.copy(), 9, 75, 75)		
    gaussian = cv2.GaussianBlur(gray.copy(), (5, 5), 0)

    bilateral_edged = cv2.Canny(bilateral, threshold1 = 75, threshold2 = 200) # was 0, 50...not sure what these numbers mean
    gaussian_edged = cv2.Canny(gaussian, threshold1 = 75, threshold2 = 200) # was 0, 50...not sure what these numbers mean
    titles = ["gray", "bilateral", "gaussian"]
    functions.plot_images([gray, bilateral_edged, gaussian_edged], titles)


def edging_comparisons(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray.copy(), (5, 5), 0)
    laplacian = cv2.Laplacian(gaussian,cv2.CV_64F)
    sobelx = cv2.Sobel(gaussian,cv2.CV_64F,1,0,ksize=5)  
    sobely = cv2.Sobel(gaussian,cv2.CV_64F,0,1,ksize=5)
    canny = cv2.Canny(gaussian, threshold1 = 75, threshold2 = 200) # was 0, 50...not sure what these numbers mean

    abs_lap64f = np.absolute(laplacian)
    lap_8u = np.uint8(abs_lap64f)
    abs_sobel64f = np.absolute(sobelx)
    sobx_8u = np.uint8(abs_sobel64f)
    abs_sobel64f = np.absolute(sobely)
    soby_8u = np.uint8(abs_sobel64f)

    lap_thresh = cv2.adaptiveThreshold(src = lap_8u, maxValue = 25, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 2)
    sobx_thresh = cv2.adaptiveThreshold(src = sobx_8u, maxValue = 25, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 2)
    soby_thresh = cv2.adaptiveThreshold(src = soby_8u, maxValue = 25, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 2)
    canny_thresh = cv2.adaptiveThreshold(src = canny, maxValue = 25, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 2)
    titles = ["laplacian", "sobelx", "sobely", "canny"]
    functions.plot_images([laplacian, sobelx, sobely, canny], titles)
    titles = ["laplacian", "sobelx", "sobely", "canny_tresh"]
    functions.plot_images([lap_thresh, sobx_thresh, soby_thresh, canny_thresh], titles)

def canny_comparison(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, threshold1 = 0, threshold2 = 40, apertureSize=3)
    l2_canny = cv2.Canny(blurred, threshold1 = 0, threshold2 = 80, apertureSize=3 , L2gradient=False) 
    # l2_canny = cv2.Canny(blurred, threshold1 = 0, threshold2 = 80, apertureSize=3)
    canny2 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 100, apertureSize=3)
    l2_canny2 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 160, apertureSize=3 , L2gradient=False) 
    canny3 = cv2.Canny(blurred, threshold1 = 75, threshold2 = 200, apertureSize=3, L2gradient=False)
    canny4 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 200, apertureSize=3)
    kernel = np.ones((5,5),np.uint8) # original was 15x15. 5x5 is working well right now. 
    dilated_canny = cv2.dilate(canny, kernel, iterations=1) 
    edged = [gray, canny, l2_canny, canny2, l2_canny2, canny3, canny4]
    dilated = functions.process_several(images = edged, function = cv2.dilate, kernel = kernel)
    # dilated_l2 = cv2.dilate(l2_canny, kernel, iterations=1)
    # functions.plot_images([gray, dilated_canny, dilated_l2, canny2, l2_canny2])
    functions.plot_images(dilated)
    # functions.plot_images([gray, canny, l2_canny, canny2, l2_canny2, canny3, canny4])


def downsized_canny(image):
    image = functions.standard_resize(image, 100.0, return_ratio = False)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, threshold1 = 0, threshold2 = 40, apertureSize=3)
    l2_canny = cv2.Canny(blurred, threshold1 = 0, threshold2 = 80, apertureSize=3 , L2gradient=False) 
    # l2_canny = cv2.Canny(blurred, threshold1 = 0, threshold2 = 80, apertureSize=3)
    canny2 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 100, apertureSize=3)
    l2_canny2 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 160, apertureSize=3 , L2gradient=False) 
    canny3 = cv2.Canny(blurred, threshold1 = 75, threshold2 = 200, apertureSize=3, L2gradient=False)
    canny4 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 200, apertureSize=3)
    kernel = np.ones((5,5),np.uint8) # original was 15x15. 5x5 is working well right now. 
    # dilated_canny = cv2.dilate(canny, kernel, iterations=1) 
    edged = [gray, canny, l2_canny, canny2, l2_canny2, canny3, canny4]
    functions.plot_images(edged)
    dilated = functions.process_several(images = edged, function = cv2.dilate, kernel = kernel)
    # dilated_l2 = cv2.dilate(l2_canny, kernel, iterations=1)
    # functions.plot_images([gray, dilated_canny, dilated_l2, canny2, l2_canny2])
    functions.plot_images(dilated)


def canny(image):
    # image = functions.standard_resize(image, 100.0, return_ratio = False)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0) # was this...seems like 3,3 might be better/more consistent for higher thresholds esp., specifically for downsized images
    # blurred = functions.thresholding(image)

    # canny100 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 100, apertureSize=3)
    canny100 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 80, apertureSize=3)
    canny120 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 120, apertureSize=3)
    canny140 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 140, apertureSize=3)
    canny160 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 160, apertureSize=3)
    canny180 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 180, apertureSize=3 , L2gradient=False) 
    canny200 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 200, apertureSize=3, L2gradient=False)
    canny220 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 220, apertureSize=3)

    edged = [gray, blurred, canny100, canny120, canny140, canny160, canny180, canny200, canny220]
    titles = ["gray", "blurred", "canny100", "canny120", "canny140", "canny160", "canny180", "canny200", "canny220"]
    return edged, titles


def canny_thresh(image):
    # image = functions.standard_resize(image, 100.0, return_ratio = False)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0) # was this...seems like 3,3 might be better/more consistent for higher thresholds esp., specifically for downsized images
    # blurred = functions.thresholding(image)

    canny100 = cv2.Canny(blurred, threshold1 = 75, threshold2 = 100, apertureSize=3)
    canny120 = cv2.Canny(blurred, threshold1 = 75, threshold2 = 120, apertureSize=3)
    canny140 = cv2.Canny(blurred, threshold1 = 75, threshold2 = 140, apertureSize=3)
    canny160 = cv2.Canny(blurred, threshold1 = 75, threshold2 = 160, apertureSize=3)
    canny180 = cv2.Canny(blurred, threshold1 = 75, threshold2 = 180, apertureSize=3 , L2gradient=False) 
    canny200 = cv2.Canny(blurred, threshold1 = 75, threshold2 = 200, apertureSize=3, L2gradient=False)
    canny220 = cv2.Canny(blurred, threshold1 = 75, threshold2 = 220, apertureSize=3)

    edged = [gray, blurred, canny100, canny120, canny140, canny160, canny180, canny200, canny220]
    titles = ["THRESH", "blurred", "canny100", "canny120", "canny140", "canny160", "canny180", "canny200", "canny220"]
    return edged, titles


def canny_L2(image):
    # image = functions.standard_resize(image, 100.0, return_ratio = False)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny100 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 100, apertureSize=3, L2gradient=True)
    canny120 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 120, apertureSize=3, L2gradient=True)
    canny140 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 140, apertureSize=3, L2gradient=True)
    canny160 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 160, apertureSize=3, L2gradient=True)
    canny180 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 180, apertureSize=3, L2gradient=True) 
    canny200 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 200, apertureSize=3, L2gradient=True)
    canny220 = cv2.Canny(blurred, threshold1 = 0, threshold2 = 220, apertureSize=3, L2gradient=True)
    edged = [gray, blurred, canny100, canny120, canny140, canny160, canny180, canny200, canny220]
    titles = ["L2", "blurred", "canny100 L2", "canny120", "canny140", "canny160", "canny180", "canny200", "canny220"]
    return edged, titles
    

def downsized_canny_detection(image):
    downsized, ratio = functions.standard_resize(image, new_width = 100.0)
    edged, titles = canny(downsized)
    originals = utility.image_array(downsized, array_length = len(edged))
    # detected = functions.process_several(images = edged, function = demo.vanilla_boxing)
    detected = functions.draw_several(images = edged, drawing_images = originals, function = demo.vanilla_box_drawing)
    functions.plot_images(edged, titles)
    functions.plot_images(detected,titles)


def downsized_canny_CI(image):
    downsized, ratio = functions.standard_resize(image, new_width = 250.0)
    # edged, titles = canny_thresh(downsized)
    edged, titles = canny(downsized)
    originals = utility.image_array(downsized, array_length = len(edged))
    closed_inverted = functions.process_several(images = edged, function = functions.closed_inversion)
    # detected = functions.process_several(images = closed_inverted, function = demo.vanilla_boxing)
    detected = functions.draw_several(images = closed_inverted, drawing_images = originals, function = demo.vanilla_box_drawing)

    functions.plot_images(edged,titles)
    # functions.plot_images(closed_inverted, titles)
    functions.plot_images(detected,titles)
    # return detected

def downsized_canny_dilated(image):
    downsized, ratio = functions.standard_resize(image)
    edged, titles = canny(downsized)
    originals = utility.image_array(downsized, array_length = len(edged))
    
    kernel = np.ones((5,5),np.uint8) # original was 15x15. 5x5 is working well right now. 
    dilated = functions.process_several(images = edged, function = cv2.dilate, kernel = kernel)
    detected = functions.draw_several(images = dilated, drawing_images = originals, function = demo.vanilla_box_drawing)

    functions.plot_images(dilated, titles)
    functions.plot_images(detected,titles)

def detection(image):
    downsized = functions.standard_resize()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, threshold1 = 0, threshold2 = 140, apertureSize=3, L2gradient=True)
    closed_inversion = functions.closed_inversion(edged)
    boxes = functions.alternateRectMethod(closed_inversion)
    return boxes



    
def morph_comparison(image):
    return 2


if __name__ == '__main__':
    # image = cv2.imread(sys.argv[1])
    # orig = image.copy()

    # image = functions.standard_resize(image, new_width = 1000.0)
    # edged = functions.text_edging(image)
    # edged = functions.page_edging(image)
    # downsized, _ = functions.standard_resize(edged, new_width = 100.0)
    # functions.plot_images([downsized])
    # downsized, _ = functions.standard_resize(image, new_width = 100.0)
    # canny_comparison(downsized)

    files = ["pics/demo/IMAG0603.jpg", "pics/demo/IMAG0604.jpg", "pics/demo/IMAG0605.jpg", "pics/demo/IMAG0606.jpg", "pics/demo/IMAG0607.jpg", "pics/demo/IMAG0608.jpg", "pics/demo/IMAG0611.jpg", "pics/demo/IMAG0612.jpg"]
    # files = ["pics/forms/sample5.jpg", "pics/forms/sample2.jpg", "pics/forms/sample3.jpg", "pics/forms/sample4_4.jpg", "pics/forms/sample9.jpg", "pics/forms/sample11.jpg", "pics/forms/sample8.jpg", "pics/forms/sample12.jpg"]
    # files = ["pics/forms/sample5.jpg", "pics/forms/sample2.jpg", "pics/forms/sample4_4.jpg", "pics/forms/sample11.jpg", "pics/forms/sample12.jpg", "pics/forms/sample10.jpg"]

    originals, images = utility.image_reading(files)

    # res = functions.process_several(images,downsized_canny)
    # res = functions.process_several(images,downsized_canny_finetune)
    # res = functions.process_several(images,downsized_canny_dilated)
    res = functions.process_several(images,downsized_canny_CI)
    # res = functions.process_several(images,downsized_canny_detection)
    # result = functions.process_several()
