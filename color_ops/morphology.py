import cv2
import numpy as np
import sys
import functions
import imutils
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive

if __name__ == '__main__':

    # these constants are carefully picked
    MORPH = 9
    CANNY = 84
    HOUGH = 25

    image = cv2.imread(sys.argv[1])
    orig = image.copy()

    # <---- RESIZING -----> #
    height, width = image.shape[:2]
    new_width = 100.0
    scaling_factor = new_width/width
    ratio = 1/scaling_factor
    # image = imutils.resize_new(image, scaling_factor = scaling_factor)
    # <---- RESIZING -----> #


    # <---- FUNCTIONS.PY PROCESS -----> #
    edged = functions.colorOps(image)
    closed_inversion = functions.closed_inversion(edged)
    images = [edged, closed_inversion]
    titles = ["edged", "closed_inversion"]
    # functions.plot_images(images, titles)


    # <---- FUNCTIONS.PY PROCESS -----> #

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bordered = cv2.copyMakeBorder(blurred, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[0, 0, 0])		 # added this 11/2/17. Trying to work with document occlusion. 
    # cv2.GaussianBlur(gray, (3,3), 0, gray) # this is the original from this process

    # this is to recognize white on white
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
    # dilated = cv2.dilate(gray, kernel)

    kernel = np.ones((15,15),np.uint8)

    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(blurred,kernel,iterations = 1)
    dilate_bordered = cv2.dilate(bordered,kernel,iterations = 1)
    # dilate_edged = cv2.Canny(dilation, 0, CANNY, apertureSize = 3) # this is the original from this process


    # <---- RESIZING -----> #
    height, width = dilation.shape[:2]
    new_width = 100.0
    scaling_factor = new_width/width
    ratio = 1/scaling_factor
    dilation_small = imutils.resize_new(dilation, scaling_factor = scaling_factor)
    bd_small = imutils.resize_new(dilate_bordered, scaling_factor = scaling_factor)
    # <---- RESIZING -----> #
    dsb = cv2.copyMakeBorder(dilation_small, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])		 # added this 11/2/17. Trying to work with document occlusion. 

    # dilate_large = cv2.Canny(dilation, threshold1 = 75, threshold2 = 200) # this doesn't work very well following dilation (for both original and resized)
    dilate_large = cv2.Canny(blurred, 0, CANNY, apertureSize=3)
    dilate_edged = cv2.Canny(dilation_small, threshold1 = 75, threshold2 = 200) # this doesn't work very well following dilation (for both original and resized)
    bds_edged = cv2.Canny(bd_small, threshold1 = 75, threshold2 = 200) 
    dsb_edged = cv2.Canny(dsb, threshold1 = 75, threshold2 = 200)

    closedEdges = cv2.morphologyEx(bds_edged, cv2.MORPH_CLOSE, kernel = np.ones((5, 11))) # this potentially helps close Canny edges that are close but not quite touching
    closedInvert = cv2.bitwise_not(src = closedEdges.copy())

    titles = ["orig", "dilate_edged", "border->dilate->shrink", "closed_invert", "dilate->shrink->border"]
    images = [orig, dilate_edged, bds_edged, closedInvert, dsb_edged]

    # images = [blurred, opening, dilation, dilate_edged] # it looks like 'opening has a postive effect on text readability 
    # titles = ["Orig", "open", "dilation", "dilate_edged"]

    functions.plot_images(images, titles)

    edges = cv2.Canny(blurred, 0, CANNY, apertureSize=3) # this is the original from this process. Below is what's working in other conditions 
    # edges = cv2.Canny(blurred, threshold1 = 75, threshold2 = 200) # was 0, 50...not sure what these numbers mean

    blackhat = cv2.morphologyEx(edges, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(edges, cv2.MORPH_TOPHAT, kernel)
    gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel) # this looks good!! (input original image or the canny edged photo)
    # gradient = cv2.morphologyEx(dilate_edged, cv2.MORPH_GRADIENT, kernel)  
    # gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

    height, width = gradient.shape[:2]
    new_width = 100.0
    scaling_factor = new_width/width
    ratio = 1/scaling_factor
    gradient_small = imutils.resize_new(gradient, scaling_factor = scaling_factor)


    gradient_edged = cv2.Canny(gradient_small, threshold1 = 75, threshold2 = 200) 
    # gradient_edged = cv2.Canny(gradient, 0, CANNY, apertureSize=3)

    # images = [edges, blackhat, tophat, gradient] # none of these seem particularly useful besides "gradient"
    # titles = ["edged", "blackhat", "tophat", "gradient"]

    binarized = threshold_adaptive(gradient, 251, offset = 10)
    final = binarized.astype("uint8") * 255

    images = [edges, gradient, binarized, gradient_edged] # none of these seem particularly useful besides "gradient"
    titles = ["edged", "gradient", "gradient_bin", "gradient_edged"]

    # functions.plot_images(images, titles)

