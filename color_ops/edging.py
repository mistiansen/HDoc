import cv2
import sys
import numpy as np
import argparse
import imutils
import math
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive


def text_edging(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, threshold1 = 0, threshold2 = 80) # was 0, 50...not sure what these numbers mean
    kernel = np.ones((5,5),np.uint8) # original was 15x15. 5x5 is working well right now. 
    dilated = cv2.dilate(edged, kernel, iterations=1) # THIS LOOKS GOOD ON ALL but s4!! It erodes too much though. Losing some of document.
    closed = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel = np.ones((5, 11))) # this potentially helps close Canny edges that are close but not quite touching
    return closed

def downsized_text_edging(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, threshold1 = 0, threshold2 = 60) # 0, 80 or 0, 100 I guess work best
    # kernel = np.ones((5,5),np.uint8) # original was 15x15. 5x5 is working well right now. 
    # dilated = cv2.dilate(edged, kernel, iterations=1) # THIS LOOKS GOOD ON ALL but s4!! It erodes too much though. Losing some of document.
    # closed = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel = np.ones((5, 11))) # this potentially helps close Canny edges that are close but not quite touching
    # return closed
    return edged

def text_blobbing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = cv2.Canny(blurred, threshold1 = 0, threshold2 = 80) # was 0, 50...not sure what these numbers mean 
    kernel = np.ones((5,5),np.uint8) # original was 15x15. 5x5 is working well right now. 
    blobbed = cv2.dilate(edged, kernel, iterations=1) # THIS LOOKS GOOD ON ALL but s4!! It erodes too much though. Losing some of document.
    # blobbed = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel = np.ones((1, 1))) # this potentially helps close Canny edges that are close but not quite touching
    # blobbed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel = np.ones((1, 1))) # this potentially helps close Canny edges that are close but not quite touching
    return blobbed

# could just use 1 method with the parameters set at the method call (defined here)
def page_edging(image, thresh1, thresh2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # edged = cv2.Canny(blurred, threshold1 = 75, threshold2 = 200) # RECENT EDIT: ORIGINAL WAS 75,200. Trying higher thresholds below.  
    edged = cv2.Canny(blurred, threshold1 = thresh1, threshold2 = thresh2)
    return edged    

def orig_page_edging(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # edged = cv2.Canny(blurred, threshold1 = 0, threshold2 = 200) # was 75, 200...these seem to work better 
    # edged = cv2.Canny(blurred, threshold1 = 75, threshold2 = 200) # RECENT EDIT: ORIGINAL WAS 75,200. Trying higher thresholds below.  
    edged = cv2.Canny(blurred, threshold1 = 75, threshold2 = 220)
    return edged

def new_page_edging(image, thresh1, thresh2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # edged = cv2.Canny(blurred, threshold1 = 75, threshold2 = 200) # RECENT EDIT: ORIGINAL WAS 75,200. Trying higher thresholds below.  
    edged = cv2.Canny(blurred, threshold1 = thresh1, threshold2 = thresh2)
    return edged    

def auto_edging(image, sigma = 0.33):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

