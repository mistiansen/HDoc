import sys
import cv2
import argparse
import numpy as np
import functions
from matplotlib import pyplot as plt

def text_regions(image):
    vis = image.copy()
    # image = cv2.imread('../service_form.png', 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # mser = cv2.MSER() # was working but now isnt? version change?
    mser = cv2.MSER_create() # trying this

    # regions = mser.detect(image, None)
    regions = mser.detectRegions(image, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))

    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    #this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(image, image, mask=mask)       

    # text_only = functions.standard_resize(text_only)
    # text_only = cv2.bitwise_not(src = text_only.copy())

    return text_only

if __name__ == '__main__':
    image = cv2.imread(sys.argv[1])
    text_only = text_regions(image)
    functions.plot_images([text_only])

