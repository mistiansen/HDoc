import cv2
import sys
import numpy as np
import argparse
import rect
import imutils
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive
 

# <---- IMAGE COLOR, SMOOTHING, AND CANNY EDGE DETECTION PRE-PROCESSING -----> # 

def colorOps(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv2.copyMakeBorder(blurred, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=[0, 0, 0])		 # added this 11/2/17. Trying to work with document occlusion. 
    edged = cv2.Canny(blurred, threshold1 = 75, threshold2 = 200) # was 0, 50...not sure what these numbers mean
    return edged

def closed_inversion(image):
    closedEdges = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel = np.ones((5, 11))) # this potentially helps close Canny edges that are close but not quite touching
    closedInvert = cv2.bitwise_not(src = closedEdges.copy())
    kernel = np.ones((5,5),np.uint8) # original was 15x15. 5x5 is working well right now. 
    erosion = cv2.erode(closedInvert, kernel, iterations=1) # THIS LOOKS GOOD ON ALL but s4!! It erodes too much though. Losing some of document.
    return erosion # THIS WORKS WELL FOR PRETTY MUCH ALL DOCUMENTS SO FAR (except sample4)

# <---- DOCUMENT DETECTION METHODS -----> #

def fetch_largest_contours(image, n_largest = 5):
    ims, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # trying CHAIN_APPROX_SIMPLE...DOES THE SAME THING
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:n_largest] # return the n largest contours
    return contours


def contour_method(image):
    contours = fetch_largest_contours(image)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) # original line...0.02 * p

        if len(approx) == 4:
            target = approx
            break

    # detected = cv2.drawContours(image, [target], -1, (0, 255, 0), 1)
    print "printing target: " + str(target)
    return target # so the length of this will always be 4, because of the check above. That is not the case for the hullMethod


def minRectMethod(image):
    contours = fetch_largest_contours(image)
    rect = cv2.minAreaRect(contours[0])
    print "this is contour area " + str(cv2.contourArea(contours[0]))
    print "this is image area " + str(image.shape[0] * image.shape[1])

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # this is just so can see the green bounding box
    detected = cv2.drawContours(image.copy(),[box],0,(0,255,0),1)
    # plot_images([detected])
    return box

def alternateRectMethod(image):
    contours = fetch_largest_contours(image)
    detected = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB) # this is just so can see the green bounding box
    boxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        detected = cv2.drawContours(detected,[box],0,(0,255,0),1) # drawing all the boxes
        boxes.append(box)
    plot_images([detected])
    return boxes

def hullMethod(image): # could also potentially make a comparable method but using minAreaRect instead of approxPolyDP
    contours = fetch_largest_contours(image)
    simplified_contours = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        simplified_contour = cv2.approxPolyDP(hull, 0.1*cv2.arcLength(hull, True), True)
        if len(simplified_contour) == 4:
            simplified_contours.append(simplified_contour)
    detected = cv2.drawContours(image.copy(),simplified_contours,0,(0,255,0),1)
    # plot_images([detected])
    return simplified_contours # problem is this won't work with a "len == 4 check when occlusion is present. Sample problems as the other method."

def hullRectMethod(image):
    contours = fetch_largest_contours(image)
    simplified_contours = []
    detected = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        detected = cv2.drawContours(detected,[box],0,(0,255,0),1)
        simplified_contours.append(box)
    plot_images([detected])
    return simplified_contours


def finalize(image, points, ratio): 
    print "printing points in finalize " + str(points)
    warped = imutils.four_point_transform(image = image.copy(), pts = points.reshape(4, 2) * ratio) # for use with minAreaRect result
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gaussian_threshold = cv2.adaptiveThreshold(src = blur, maxValue = 25, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 2)
    return gaussian_threshold

def standard_resize(image):
        # <---- RESIZING -----> #
    height, width = image.shape[:2]
    new_width = 100.0
    scaling_factor = new_width/width
    ratio = 1/scaling_factor
    return imutils.resize_new(image, scaling_factor = scaling_factor)
    # <---- RESIZING -----> #

def rotate(image, angle): # leaving center as the midpoint and scale as 1 by default
    h, w = image.shape[:2]
    (centerX, centerY) = (w/2, h/2)
    rotation_matrix = cv2.getRotationMatrix2D(center = (centerX, centerY), angle = angle, scale = 1.0) 
    cos = np.abs(rotation_matrix[0,0]) # this is just the position of alpha in the rotation matrix
    sin = np.abs(rotation_matrix[0,1]) # this is just the position of beta in the rotation matrix
    new_width = int((h * sin) + (w * cos))
    new_height = int((h * cos) + (w * sin))
    rotation_matrix[0, 2] += (new_width / 2) - centerX
    rotation_matrix[1, 2] += (new_height / 2) - centerY
    rotated = cv2.warpAffine(src = image, M = rotation_matrix, dsize = (new_width, new_height))
    return rotated
    

if __name__ == '__main__':
    image = cv2.imread(sys.argv[1])


    dilation_canny(image)
    # rotate(image, 20)
    # occlusion_demo(image)
    # original_demo(image)
    # hull_attempt(image)
    # alternate_rect_attempt(image)
    # new_docs(image)
    # blur_comparison(image)
    # edging_comparisons(image)

