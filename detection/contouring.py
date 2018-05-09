
import cv2
import sys
import numpy as np
import argparse
import imutils
import math
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive


# <---- DOCUMENT DETECTION METHODS -----> #

def fetch_largest_contours(image, n_largest = 5):
    ims, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # trying CHAIN_APPROX_SIMPLE...DOES THE SAME THING
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:n_largest] # return the n largest contours
    # print( type(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True) # return the n largest contours
    # for i, contour in enumerate(contours):
    #     print( "got contour area " + str(cv2.contourArea(contour)) + " for contour " + str(i)
    return contours[:n_largest]


def contour_method(image):
    contours = fetch_largest_contours(image)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) # original line...0.02 * p

        if len(approx) == 4:
            target = approx
            break

    # detected = cv2.drawContours(image, [target], -1, (0, 255, 0), 1)
    print("printing target: " + str(target))
    return target # so the length of this will always be 4, because of the check above. That is not the case for the hullMethod


def minRectMethod(image):
    contours = fetch_largest_contours(image)
    rect = cv2.minAreaRect(contours[0])
    print("this is contour area " + str(cv2.contourArea(contours[0])))
    print("this is image area " + str(image.shape[0] * image.shape[1]))

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # this is just so can see the green bounding box
    detected = cv2.drawContours(image.copy(),[box],0,(0,255,0),1)
    # plot_images([detected])
    return box

def pureRectMethod(image):
    contours = fetch_largest_contours(image)
    detected = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB) # this is just so can see the green bounding box
    image_height = image.shape[0]
    image_width = image.shape[1]
    image_area = image_width * image_height
    boxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box_area = cv2.contourArea(box)
        w, h = rect[1][:2] 
        w = w + 3 # this is to recover some loss from the morphological operations
        h = h + 3 # this is to recover some loss from the morphological operations
        new_rect = (rect[0], (w,h), rect[2])
        box = cv2.boxPoints(new_rect)
        box = np.int0(box)
        boxes.append(box)
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