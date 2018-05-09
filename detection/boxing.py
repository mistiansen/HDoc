import cv2
import sys
import numpy as np
import argparse
import imutils
import math
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive

import coloring
import contouring
import utility


# <------ MAIN BOXING METHDOS --------> #

def box_generation(image):
    contours = contouring.fetch_largest_contours(image)
    # detected = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB) # this is just so can see the green bounding box
    image_width, image_height = image.shape[1], image.shape[0]
    image_area = image_width * image_height
    boxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        rect_angle = rect[2]
        # if rect_center_in_quadrant(rect=rect, image_width=image_width, image_height=image_height) or  45 < abs(rect_angle) < 90:
        if rect_center_in_quadrant(rect=rect, image_width=image_width, image_height=image_height):
            # print("FAILED RECT QUADRANT CHECK")
            continue # WAS THIS COMMENTED OUT?!
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box_area = cv2.contourArea(box)
        if (box_area > (0.88 * image_area) or box_area < 0.10 * image_area or box_in_image_half(box, image_width, image_height)):
        # if (box_area > (0.90 * image_area) or box_area < 0.10 * image_area):
            # print("box rejected in alternateRectMethod. Area was: " + str(box_area))
            continue
        # elif (box_area > (0.75 * image_area) and (abs(rect[2]) == 0 or abs(rect[2]) == 90)):
            # print( "WNTING TO DISCARD LARGE, ALIGNED BOX " + str(box_area/image_area * 100) + "% of image area"
        else:
            boxes.append(box)
    return boxes

# if a box is axis-aligned and too close to the image's edges, it is most likely a false positive.
def box_filtration(boxes, image_width, image_height):
    filtered = []
    for box in boxes:
        rect = cv2.minAreaRect(box)
        angle = rect[2]
        # print("angle in box_filtration " + str(abs(angle)))
        if abs(angle) == 0 or abs(angle) == 90:
            xmin, xmax, ymin, ymax = box_extrema(box)
            # if -2 < xmin < 2 and (-2 < ymin < 2 or image_height - 0.02 * image_height <= ymax <= 1.02 * image_height):
            if -2 < xmin < 2 and (-2 < ymin < 2 or image_height - 2 <= ymax <= image_height + 2):
                # print("FILTERED A BOX FROM THE TOP")
                continue
            # elif image_width - (0.02 * image_width) <= xmax <= 1.02 * image_width and ( -2 <= ymin <= 2 or image_height - (0.02 * image_height) <= ymax <= (1.02 * image_height)):
            elif image_width - 2 <= xmax <= image_width + 2 and ( -2 <= ymin <= 2 or image_height - 2 <= ymax <= image_height + 2):
                # print("FILTERED A BOX FROM THE BOTTOM")
                continue
            else:
                filtered.append(box)
        else:
            filtered.append(box)
    return filtered

# <------ END MAIN BOXING METHDOS --------> #

# <------ BOXING SUPPORT METHDOS --------> #

def all_boxes(image):
    contours = contouring.fetch_largest_contours(image)
    # detected = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB) # this is just so can see the green bounding box
    boxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)
    return boxes

def box_in_image_half(box, image_width, image_height): # returns true if the contour/box resides entirely in one half of an image
    print(box.shape)
    image_midX = image_width / 2
    image_midY = image_height / 2
    xMin, xMax, yMin, yMax = box_extrema(box)
    if(xMax < image_midX or xMin > image_midX or yMax < image_midY or yMin > image_midY):
        return True
    else:
        return False

def box_extrema(box):
    xSorted = box[np.argsort(box[:, 0]), :] # this means sort by rows I think (row, col). Each row is a point. 
    ySorted = box[np.argsort(box[:, 1]), :]
    xMin = xSorted[0][0]
    xMax = xSorted[3][0]
    yMin = ySorted[0][1]
    yMax = ySorted[3][1]
    return xMin, xMax, yMin, yMax

def rect_center_in_quadrant(rect, image_width, image_height):
    x_qtr, y_qtr = image_width / 4, image_height / 4
    rect_center_x, rect_center_y = rect[0][0], rect[0][1]
    if rect_center_x < x_qtr or rect_center_x > 3 * x_qtr or rect_center_y < y_qtr or rect_center_y > 3 * y_qtr:
        return True
    else:
        return False

def full_overlap(box1, box2):
    xMin1, xMax1, yMin1, yMax1 = box_extrema(box1)
    xMin2, xMax2, yMin2, yMax2 = box_extrema(box2)
    if xMin1 > xMin2 and xMax1 < xMax2 and yMin1 > yMin2 and yMax1 < yMax2:
        return True
    elif xMin2 > xMin1 and xMax2 < xMax1 and yMin2 > yMin1 and yMax2 < yMax1:
        return True
    else:
        return False

def max_points(points):
    xSorted = points[np.argsort(points[:, 0]), :] # this means sort by rows I think (row, col). Each row is a point. 
    ySorted = points[np.argsort(points[:, 1]), :]
    xMin = xSorted[0][0]
    xMax = xSorted[-1][0]
    yMin = ySorted[0][1]
    yMax = ySorted[-1][1]
    return xMin, xMax, yMin, yMax
    # max = np.array([[xMin, yMax], [xMin, yMin], [xMax, yMin], [xMax, yMax]])
    # return merged
        
def merge_boxes(boxes):
    rows = len(boxes) * 4
    if not boxes:
        return boxes
    elif len(boxes) == 1:
        return boxes[0]
    else:
        points = np.vstack(boxes) # stacks all of the list elements (which are numpy arrays) into a numpy array
        # print("merging points: " + str(points))
        xSorted = points[np.argsort(points[:, 0]), :] # this means sort by rows I think (row, col). Each row is a point. 
        ySorted = points[np.argsort(points[:, 1]), :]
        xMin = xSorted[0][0]
        xMax = xSorted[rows - 1][0]
        # xMax = xSorted[-1][0]
        yMin = ySorted[0][1]
        yMax = ySorted[rows - 1][1]
        # yMax = ySorted[-1][1]
        merged = np.array([[xMin, yMax], [xMin, yMin], [xMax, yMin], [xMax, yMax]]) # looks like OpenCV orders bottom left, top left, top right, bottom right
    # print("merged is " + str(merged))
    return merged 

def box_crop(image, box):
    xmin, xmax, ymin, ymax = max_points(box)
    points = threshold_negative_points([xmin, xmax, ymin, ymax])
    xmin, xmax, ymin, ymax = points
    cropped = image[ymin:ymax+1, xmin:xmax+1].copy() # copy because don't want to alter the passed-in image
    return cropped # copy because don't want to alter the passed-in image

def threshold_negative_points(points):
    for i in range(len(points)):            
        if points[i] < 0:
            points[i] = 0
    return points

def boxes_comparison(image, boxes):
    #compare contour[0] to image
    # compare contour[0] to contour[1]
    image_area = image.shape[0] * image.shape[1]
    print("image area in boxes_comparison is " + str(image_area))
    areas = []
    for i, box in enumerate(boxes):
        box_area = cv2.contourArea(box)
        print("got box area : " + str(box_area) + " for box " + str(i) + " in boxes_comparison")
        print("that is " + str((box_area/image_area) * 100) + "% of total image area")

def boxes_from_edged(image, edging_function, **edging_args):
    edged = edging_function(image, **edging_args) # perform edge detection on the input image according to the passed in edging function
    closed_invert = coloring.closed_inversion(edged) # perform closed inversion (morphological closing + color inversion) on the edged image
    boxes = box_generation(closed_invert) # generate bounding boxes from the closed inversion output
    boxes = box_filtration(boxes, image.shape[1], image.shape[0]) # filter bounding boxes
    merged = merge_boxes(boxes)
    return merged, boxes, edged, closed_invert

def edged_to_boxes(image, edging_function, **edging_args):
    edged = edging_function(image, **edging_args) # perform edge detection on the input image according to the passed in edging function
    closed_invert = coloring.closed_inversion(edged) # perform closed inversion (morphological closing + color inversion) on the edged image
    boxes = box_generation(closed_invert) # generate bounding boxes from the closed inversion output
    boxes = box_filtration(boxes, image.shape[1], image.shape[0]) # filter bounding boxes
    merged = merge_boxes(boxes)
    return merged

def generate_boxes(image):
    boxes = box_generation(image)
    boxes = box_filtration(boxes, image.shape[1], image.shape[0])
    merged = merge_boxes(boxes)
    return merged 


def vanilla_box_drawing(image, drawing_image):
    boxes = box_generation(image)
    # functions.boxes_comparison(image, boxes)
    # boxes = functions.all_boxes(image)
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # this is just so can see the green bounding box

    boxes = box_filtration(boxes, image.shape[1], image.shape[0])
    box = []
    # detected = drawing_image
    if not boxes:
        # print "NO BOXES FOUND"
        return drawing_image
    else:   
        for box in boxes:
            # print box
            # # print type(box)
            drawing_image = cv2.drawContours(drawing_image,[box],0,(0,255,0),1)
    return drawing_image
    

def vanilla_boxing(image):
    boxes = box_generation(image)
    boxes_comparison(image, boxes)
    # boxes = functions.all_boxes(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # this is just so can see the green bounding box

    box = []
    detected = image
    if not boxes:
        print("NO BOXES FOUND")
    else:   
        for box in boxes:
            # # print box
            detected = cv2.drawContours(detected,[box],0,(0,255,0),1)
            # box = boxes[0]
            # final = functions.finalize(orig.copy(), box, ratio)
    # functions.plot_images([image, detected], ["edged", "detected"])
    return detected