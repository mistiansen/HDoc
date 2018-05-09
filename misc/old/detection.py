import cv2
import numpy as np
import argparse
import rect
import imutils
import functions
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image file")
args = vars(ap.parse_args())

image = cv2.imread(args["image"], 1)
orig = image.copy()

height, width = image.shape[:2]
new_width = 100.0
scaling_factor = new_width/width
# ratio = image.shape[0] / 500.0
# ratio = image.shape[1] / 100.0
# image = imutils.resize(image, height = 100)
ratio = 1/scaling_factor
image = imutils.resize_new(image, scaling_factor = scaling_factor)
print ratio


# convert to grayscale and blur to smooth
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# blurred = cv2.copyMakeBorder(blurred, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])		 # added this 11/2/17. Trying to work with document occlusion. 

# apply Canny Edge Detection ....look into marheldritch
edged = cv2.Canny(blurred, threshold1 = 75, threshold2 = 200) # was 0, 50...not sure what these numbers mean

closedEdges = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel = np.ones((5, 11))) # added this 11/2/17. Trying to work with document occlusion. 

# mask = np.ones(closedEdges.shape,np.uint8)
closed2 = closedEdges.copy()
# cv2.bitwise_and(closed2,closed2,mask)
cv2.bitwise_not(closedEdges, closed2)

images = [image, edged]
images = [edged, closedEdges]
images = [closedEdges, closed2]

functions.plot_images(images)


# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
#(contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # original line ('too many values to unpack')
ims, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # modified line with CHAIN_APPROX_NONE
# ims, contours, hierarchy = cv2.findContours(closedEdges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # trying CHAIN_APPROX_SIMPLE...DOES THE SAME THING
# ims, contours, hierarchy = cv2.findContours(closed2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # trying CHAIN_APPROX_SIMPLE...DOES THE SAME THING
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] # added the [:5] per the kickass scanner example

# get approximate contour

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True) # original line...0.02 * p

    if len(approx) == 4:
        target = approx
        break

detected = cv2.drawContours(image, [target], -1, (0, 255, 0), 1)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] # added the [:5] per the kickass scanner example
rect = cv2.minAreaRect(contours[0])
# print rect
box = cv2.boxPoints(rect)

print box
box = np.int0(box)
# img = np.zeros((3500,3500,3), np.uint8)

# detected = cv2.drawContours(image.copy(),[box],0,(0,255,0),1)

# <----- WARPING AND THRESHOLDING -------> #

warped = imutils.four_point_transform(image = orig, pts = target.reshape(4, 2) * ratio) # this is the original. Works on sample4.jpg
# warped = imutils.four_point_transform(image = orig, pts = box.reshape(4, 2) * ratio) # for use with minAreaRect result
# warped = imutils.four_point_transform(image = orig.copy(), pts = box.reshape(4, 2) * ratio) # for use with minAreaRect result
 
# convert the warped image to grayscale, then threshold it

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped, 251, offset = 10)
warped = warped.astype("uint8") * 255

images = [detected, orig.copy(), warped]

functions.plot_images(images)
