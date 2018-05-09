import cv2
import sys
import numpy as np
import argparse
import imutils
import math
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive


def perspective_transform(image, points, ratio = None): 
    # print( "print(ing points in finalize " + str(points)
    points = np.array(list(points)) # added this 4/17/18 to keep the idea of points, corners as tuples consistent. 
    
    height, width = image.shape[0], image.shape[1]
    if ratio:
        points = points.reshape(4,2) * ratio
    else:
        points = points.reshape(4,2)
    # points = imutils.boundary_coords(points, width, height)
    warped = imutils.four_point_transform(image = image.copy(), pts = points) # for use with minAreaRect resul
    return warped

# <---- RESIZING -----> #

def standard_resize(image, new_width = 100.0, return_ratio = True):
    height, width = image.shape[:2]
    scaling_factor = new_width/width
    ratio = 1/scaling_factor
    resized = resize_new(image, scaling_factor = scaling_factor)
    if return_ratio:
        return resized, ratio
    else:
        return resized


def resize(image, height):
    # opencv ordering is height, width
    r = 100.0 / image.shape[1] 
    dim = (100, int(image.shape[0] * r)) # dim as (width, height)?
    
    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized


def resize_new(image, scaling_factor):
    # resized = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation = cv2.INTER_AREA)
    height = int(scaling_factor * image.shape[0])
    width = int(scaling_factor * image.shape[1])
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized

# <---- END RESIZING -----> #

# <---- ROTATION -----> #

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