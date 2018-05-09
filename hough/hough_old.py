import os
import cv2
import math
import numpy as np
import sys
from matplotlib import pyplot as plt
import imutils
import functions
import decimal
import utility
import time
from scipy.spatial import distance

import edging
import reshape
import coloring
import boxing

def edged_gradient(image, kernel_size = (15,15)):
    kernel = np.ones(kernel_size,np.uint8)
    dilation = cv2.dilate(image, kernel, iterations = 1) # iterations was 1
    dilated_edged = cv2.Canny(dilation, 0, 80, apertureSize = 3) # done in this order because the dilation first gets rid of the text, then do edge detection
    gradient = cv2.morphologyEx(dilated_edged, cv2.MORPH_GRADIENT, kernel)  
    return gradient
    # return dilated_edged


def new_technique(image):
    original = image.copy()
    outlined = edged_gradient(image)

def computeIntersect(a, b):
    x1 = a[0][0]
    y1 = a[0][1]
    x2 = a[1][0]
    y2 = a[1][1]
    x3 = b[0][0]
    y3 = b[0][1]
    x4 = b[1][0]
    y4 = b[1][1]
    d = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

    diff_x_1 = float(x2 - x1)
    diff_x_2 = float(x4 - x3)

    if (diff_x_1 == 0):
        diff_x_1 = 0.001
    if (diff_x_2 == 0):
        diff_x_2 =0.001

    slope1 = float(y2 - y1) / diff_x_1
    slope2 = float(y4 - y3) / diff_x_2
    angle1 = math.degrees(math.atan(slope1))
    angle2 = math.degrees(math.atan(slope2))
    angle_diff = abs(angle1 - angle2)
    angle_intersection = 180 - angle_diff
    
    # if d and 80 < abs(angle_intersection) < 100:
    if d and 80 < abs(angle_intersection) < 110:
    # if d: # old version
        d = float(d)
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        # print x, y, angle_intersection
        # return int(x), int(y)
        return x, y
    else:
        return -1, -1

def standard_hough_lines(image):
    lines = []
    hough_lines = cv2.HoughLines(image, rho = 1, theta = 3.14/180, threshold = 30) # when using downsized image for corner finding, need very low threshold
    if type(hough_lines) == np.ndarray:
        for hough_line in hough_lines:
            for rho, theta in hough_line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b)) # all of these multiples were 1000
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                line = ((x1,y1), (x2,y2))
                lines.append(line)
    return lines


def standard_hough_rotation(theta, drawing_image):
    # rotation_angle = ((math.cos(theta) * 180) / math.pi) * -1
    # rotation_angle = (90 - math.degrees(theta)) * -1
    rotation_angle = (90 - math.degrees(theta)) * -1
    rotated_image = functions.rotate(drawing_image, rotation_angle)
    print("most common theta is: " + str(theta))
    print("rotation angle is: " + str(rotation_angle))
    # functions.plot_images([image, drawing_image, rotated_image], ["input image", "Std. Hough", "Rotated"])

def prob_hough(image, drawing_image):
    hough_lines = cv2.HoughLinesP(image, rho = 1, theta = 3.14/180, threshold = image.shape[1]/10, minLineLength = image.shape[:2][1]/8, maxLineGap = image.shape[:2][1]/50)  # working fairly well
    # hough_lines = cv2.HoughLinesP(image, rho = 1, theta = 3.14/180, threshold = 250, minLineLength = image.shape[:2][1]/8, maxLineGap = image.shape[:2][1]/50) #THIS IS WHAT WORKS FOR FULL SIZE IMAGES
    # hough_lines = cv2.HoughLinesP(image, rho = 1, theta = 3.14/180, threshold = 25, minLineLength = image.shape[:2][1]/8, maxLineGap = image.shape[:2][1]/50)  # working fairly well
    # lines = cv2.HoughLinesP(edges, 1,  3.14/180, HOUGH)
    # lines = cv2.HoughLinesP(edged_grad, rho = 1, theta = 3.14/180, threshold = 100, minLineLength = 150, maxLineGap = HOUGH)
    # lines = cv2.HoughLinesP(edged, rho = 1, theta = 3.14/180, threshold = 25, minLineLength = edged.shape[:2][1]/10, maxLineGap = edged.shape[:2][1]/50) 
    # lines = cv2.HoughLinesP(orig_bin.copy(), rho = 1, theta = 3.14/180, threshold = 5000, minLineLength = orig_bin.shape[:2][1]/10, maxLineGap = orig_bin.shape[:2][1]/70) 
    # lines = cv2.HoughLinesP(closed_invert, rho = 1, theta = 3.14/180, threshold = 100, minLineLength = 0, maxLineGap = 0)
    # lines = cv2.HoughLinesP(edged, rho = 1, theta = 3.14/180, threshold = 10, minLineLength = 20, maxLineGap = 10) # this seems to work with the downsized image
    lines = []
    if hough_lines is None:
        print("NO LINES FOUND")
    else:    
        print("number of lines found " + str(hough_lines.shape))
        for hough_line in hough_lines:
            for x1,y1,x2,y2 in hough_line:
                # cv2.line(drawing_image,(x1,y1),(x2,y2),(255,0,0),1)
                cv2.line(drawing_image,(x1,y1),(x2,y2),(255,255,255),1)
                line = ((x1, y1), (x2, y2))
                lines.append(line)
    return lines, drawing_image

def midpoint(line):
    pt1, pt2 = line[0], line[1]
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    mid_X = (x2 + x1) / 2
    mid_y = (y2 + y1) / 2
    return (mid_X, mid_y)

# def center_of_mass


def calculate_radians(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    denominator = (x2 - x1)
    if denominator < 0.0000001: denominator = 0.00001 # avoid dividing by 0
    slope = ((y2 - y1) * 1.0) / (denominator * 1.0)
    if slope > 100: print("FOUND LARGE SLOPE: " + str(slope))
    radians = math.atan(slope) # might want to return the radians rather than the degrees because the binning is more general
    return radians

def prob_hough_rotation(image, drawing_image):
    lines, drawn_image = prob_hough(image, drawing_image)
    radians = dict()
    for line in lines:
        pt1, pt2 = line
        x1, y1 = pt1
        x2, y2 = pt2
        radian = round(calculate_radians((x1, y1), (x2, y2)), 2)
        if radian in radians.keys(): radians[radian] = radians[radian] + 1
        else: radians[radian] = 1 
    freq_radian = max(radians, key=radians.get)
    # rotation_angle = (90 - math.degrees(freq_radian)) * -1
    rotation_angle = math.degrees(freq_radian)
    print("Most frequent radians is: " + str(freq_radian) +  "Rotation angle is: " + str(rotation_angle))
    rotated_image = functions.rotate(drawn_image.copy(), rotation_angle)
    # functions.plot_images([image, drawn_image, rotated_image], ["Input image", "HoughLinesP", "Rotated"]) 
    return rotated_image

def draw_lines(drawing_image, lines, ratio = None): # modifies the passed in image
    for line in lines:
        pt1, pt2 = line
        if ratio:
            pt1, pt2 = tuple(int(round(coord * ratio)) for coord in pt1), tuple(int(round(coord * ratio)) for coord in pt2)
        cv2.line(img = drawing_image, pt1 = pt1, pt2 = pt2, color = (255,0,0), thickness = 1)
    return drawing_image


def edging_old(image): # currently identical to functions.text_edging...keeping here for convenience
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, threshold1 = 0, threshold2 = 80, apertureSize=3, L2gradient=True)
    kernel = np.ones((5,5),np.uint8) # original was 15x15. 5x5 is working well right now. 
    dilated = cv2.dilate(edged, kernel, iterations=1) # THIS LOOKS GOOD ON ALL but s4!! It erodes too much though. Losing some of document.s  
    closed = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel = np.ones((5, 11))) # this potentially helps close Canny edges that are close but not quite touching
    result = dilated
    return result  

def image_reading(files):
    originals = []
    images = []
    for filename in files:
        image = cv2.imread(filename)
        images.append(image)
        originals.append(image.copy())
    return originals, images


# This downsizes images then processes them (downsized_text_edging) then runs Hough
def downsized_hough(originals, images): 
    downsized = functions.process_several(originals, function = reshape.standard_resize, new_width = 200.0, return_ratio = False) # breaking because now returning ratio from std. resize
    processed_downsized = functions.process_several(downsized, function = functions.downsized_text_edging)
    rotated_images = functions.draw_several(processed_downsized, drawing_images = downsized, function = prob_hough_rotation) # downsize -> process -> Hough
    return rotated_images

# This processes full size images then downsizes them then runs Hough
def hough_downsized(originals, images):
    downsized = functions.process_several(originals, function = reshape.standard_resize)
    processed = functions.process_several(images, function = functions.text_edging)    
    downsized_processed = functions.process_several(processed, function = reshape.standard_resize, new_width = 200.0, return_ratio = False)
    # rotated_images = functions.draw_several(downsized_processed, drawing_images = downsized_processed, function = prob_hough_rotation)
    rotated_images = functions.draw_several2(downsized_processed, drawing_images = downsized_processed, function = prob_hough)
    functions.plot_images(rotated_images)
    return rotated_images

def hough_fullsized(images):
    processed = functions.process_several(images, function = functions.text_edging)   
    rotated_images = functions.draw_several(processed, drawing_images = originals, function = prob_hough_rotation) 
    return rotated_images

def timing(function, **kwargs):
    start = time.clock()
    result = function(**kwargs)
    end = time.clock()
    functions.plot_images(result)
    return end - start


# wow...I'm an idiot again. Rho is always 1. Just store the most frequent 2 "vertical" thetas (how to determine this?) and 2 "horizontal" thetas. Use those to find document outline (?)
def standard_hough(image, drawing_image):
    lines = []
    count = dict()
    dst = np.ones(image.shape[:3], np.uint8)
    # dst = np.ones((h,w), np.uint8)
    # hough_lines = cv2.HoughLines(image, rho = 1, theta = 3.14/180, threshold = 75)  # this works pretty well for normal "edged" input (assuming no illustrations in image)
    # hough_lines = cv2.HoughLines(image, rho = 1, theta = 3.14/180, threshold = 950)  # need much higher threshold for "dilated" input, because is so clear
    # hough_lines = cv2.HoughLines(image, rho = 1, theta = 3.14/180, threshold = 40) # when using downsized image for corner finding, need very low threshold. This was working pretty well, but not on videos I guess
    hough_lines = cv2.HoughLines(image, rho = 1, theta = 3.14/180, threshold = 30) # when using downsized image for corner finding, need very low threshold
    if hough_lines is None:
        print("NO LINES FOUND")
    else:    
        print("number of lines found " + str(hough_lines.shape))
        for hough_line in hough_lines:
            for rho, theta in hough_line:
                # print "rho: " + str(rho) + ". theta: " + str(theta)
                if theta in count.keys(): count[theta] = count[theta] + 1
                else: count[theta] = 1
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b)) # all of these multiples were 1000
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(drawing_image,(x1,y1),(x2,y2),(255,0,0),1)
                cv2.line(dst,(x1,y1),(x2,y2),(255,0,0),1)
                line = ((x1,y1), (x2,y2))
                lines.append(line)
    # print count

    # for c in sorted(count, key=count.get, reverse=True):
    #     print c, count[c]

    # if count:
    #     theta = max(count, key=count.get)
    #     print "found theta " + str(theta)
    # standard_hough_rotation(theta, drawing_image)

    # return lines, drawing_image
    if len(lines) > 30:
        print("found too many lines in standard hough")
        return drawing_image

    '''<------------COMMENTED THIS OUT 4/18 TRYING SOMETHING ELSE ----------------->'''
    # corner_count = find_corners(lines)
    # print("Found the following corners from find_corners in standard_hough: ")
    # print(corner_count) 
    # corners = set()
    # # for group, point_set in corner_count.iteritems():
    # for group, point_set in corner_count.items():    
    #     points = np.vstack(list(point_set))
    #     x_average = int(np.mean(points[:, 0]))
    #     y_average = int(np.mean(points[:, 1]))
    #     corners.add((x_average,y_average))
    # print(corners)
    '''<------------4/18 TRYING THIS ----------------->'''
    corners = all_corners(lines)

    # print("Ended up with " + str(len(corners)) + " after in-group merging in standard_hough")
    print("Found " + str(len(corners)) + " from all_corners")

    # corners = cv2.goodFeaturesToTrack(dst, 20, 0.01, 15)
    # drawing_image = cv2.cvtColor(drawing_image, cv2.COLOR_BGR2GRAY)
    # drawing_image = np.float32(drawing_image)
    # corners = cv2.cornerHarris(drawing_image,2,3,0.04)
    # functions.plot_images([corners])
    # corners = np.int0(corners)
    
    if corners: # for my corner method
    # if type(corners) == np.ndarray:
        for corner in corners:
            x,y = corner 
            print(corner)
            # x, y = corner.ravel()
            cv2.circle(drawing_image,(x,y),2,(0,0,255),-1)

    # print corners

    # detected = image.copy()
    # boxes = functions.alternateRectMethod(detected)
    # h,w = image.shape[:2]
    # blank = np.ones((h,w,3), np.uint8)
    # if boxes:
    #     for box in boxes:
    #         detected = cv2.drawContours(blank,[box],0,(0,255,0),1)
    functions.plot_images([image, dst, drawing_image], titles = ["Canny edge detection", "Hough lines", "Hough corners"])
    # functions.plot_images([image, drawing_image, dst, detected])
    return drawing_image

# def distance(pt1, pt2):
#     dist = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
#     return dist

def dist_from_center(image_dims, point):
    center_y, center_x = image_dims[0] // 2, image_dims[1] // 2
    center = (center_x, center_y)
    dist = distance.euclidean(point, center)
    return dist

# <---------- CORNERING METHODS ------------> #

'''Slightly different than Python 2.x version because integer division now returns a float, apparently. Have to do a // b instead. '''
def find_corners(lines): # lines is output of houghlinesP
    # corners = set()
    count = dict()
    for i, lineI in enumerate(lines): # enumerate just creates a counter that you can use along with the variable in a for loop
        for lineJ in lines[i+1:]:
            x, y = computeIntersect(lineI, lineJ)
            if x >=0 and y >= 0:
                # corners.add((x,y))
                group = count.get((x//10, y//10)) # because of how integer division works in python, this essentially thresholds to groups of 10 pixels. 
                if group:
                    count.get((x//10, y//10)).add((x,y))
                else:
                    group = set([(x,y)])
                    count[(x//10, y//10)] = group # appare
    # return corners, count
    return count
    # corner_draw = image.copy()    
    # for x1,y1,x2,y2 in lines:
    #     cv2.line(corner_draw,(x1,y1),(x2,y2),(0,255,0),2) # put this code wherever want to draw corners

def all_corners(lines):
    corners = set()
    for i, lineI in enumerate(lines): # enumerate just creates a counter that you can use along with the variable in a for loop
        for lineJ in lines[i+1:]:
            x, y = computeIntersect(lineI, lineJ)
            if x >=0 and y >= 0:
                corners.add((x,y))
    corners = np.array(list(corners))
    return corners


def corner_filter(corners, image_area):
    '''
    Input: list or set of corner points (tuples)
    '''
    corners = np.array(list(corners))

    '''Sort corners on x, y'''
    xSorted = corners[np.argsort(corners[:, 0]), :] # this means sort by rows I think (row, col). Each row is a point. 
    ySorted = corners[np.argsort(corners[:, 1]), :]

    '''Get x, y global extrema'''
    xmin, xmax, ymin, ymax = xSorted[0][0], xSorted[-1][0], ySorted[0][1], ySorted[-1][1]

    '''Get xrange, yrange, midX, midY'''
    xRange = xmax - xmin
    yRange = ymax - ymin
    midX, midY = (xmin + xmax) // 2, (ymin + ymax) // 2

    '''Initialize corners and flag variables'''
    tl, tr, br, bl = (0,0), (0,0), (0,0), (0,0)
    tl_set, tr_set, br_set, bl_set = False, False, False, False
    success = False

    '''If the maximum detection area is too small, fail'''
    max_detection_size = xRange * yRange
    if max_detection_size < 0.25 * image_area:
        print("DETECTION AREA TOO SMALL")
        corners = np.array(list((tl,tr,br,bl)))
        # print(corners)
        return corners, success

    '''<---------- LEFT, RIGHT GROUPING ----------->'''
    '''Set distance threshold and form left, right groups based on distance from '''
    # x_threshold = 30 # maybe change to some percentage of image size or expected document size
    x_threshold = .40 * xRange
    leftGroup = xSorted[abs(xmin - xSorted[:,0]) < x_threshold] # leftGroup: corners/points close to xmin
    rightGroup = xSorted[abs(xmax - xSorted[:,0]) < x_threshold] # rightGroup: corners/points close to xmax

    left_top, left_bottom = np.amin(leftGroup[:,1]), np.amax(leftGroup[:,1])
    right_top, right_bottom = np.amin(rightGroup[:,1]), np.amax(rightGroup[:,1])

    '''Perform in-group grouping (form top, bottom groups for left, right groups) [can/does this replace anything?]
    I think this can improve the finding of, say, left_xMax, by finding one for top left and one for bottom left, rather than just 1 for left group'''
    xGroup_yThresh = 5
    leftTop = leftGroup[abs(left_top - leftGroup[:,1]) < xGroup_yThresh] # leftTop = points in the leftGroup whose y distance from the left_yMin is < threshold
    leftBottom = leftGroup[abs(left_bottom - leftGroup[:,1]) < xGroup_yThresh]
    rightTop = rightGroup[abs(right_top - rightGroup[:,1]) < xGroup_yThresh]
    rightBottom = rightGroup[abs(right_bottom - rightGroup[:,1]) < xGroup_yThresh]

    # print("HERE'S leftTop, leftBottom, rightTop, rightBottom")
    # print(leftTop)
    # print()
    # print(leftBottom)
    # print()
    # print(rightTop)
    # print()
    # print(rightBottom)
    # print()

    '''Get fine-grained in-group leftmost and rightmost points'''
    leftTopRightmost, leftBottomRightmost = np.amax(leftTop[:,0]), np.amax(leftBottom[:,0]) # important one 
    rightTopLeftmost, rightBottomLeftmost = np.amin(rightTop[:,0]), np.amin(rightBottom[:,0]) # important one
    leftTopLeftmost, leftBottomLeftmost = np.amin(leftTop[:,0]), np.amin(leftBottom[:,0]) # used for range checks
    rightTopRightmost, rightBottomRightmost = np.amax(rightTop[:,0]), np.amax(rightBottom[:,0]) # used for range checks

    '''Get fine-grained in-group topmost and bottommost points'''
    leftBottomTopmost, rightBottomTopmost = np.amin(leftBottom[:,1]), np.amin(rightBottom[:, 1]) # top corresponds to y min
    leftTopBottommost, rightTopBottommost = np.amax(leftTop[:,1]), np.amax(rightTop[:,1]) # bottom corresponds to y max
    
    '''Find x group value ranges'''
    left_yRange = left_bottom - left_top
    right_yRange = right_bottom - right_top
    '''Define in-group range check values'''
    leftTopXRange = leftTopRightmost - leftTopLeftmost
    leftBottomXRange = leftBottomRightmost - leftBottomLeftmost
    rightTopXRange = rightTopRightmost - rightTopLeftmost
    rightBottomXRange = rightBottomRightmost - rightBottomLeftmost

    '''<---------- TOP, BOTTOM GROUPING ----------->'''
    '''Could now have all corners set, any combo of (tl, bl) with (tr, br) '''
    # y_threshold = 15 # maybe change to some percentage of image size or expected document size. 4/19/18 was 5
    y_threshold = 0.25 * yRange 
    topGroup = ySorted[abs(ySorted[0,1] - ySorted[:,1]) < y_threshold] # lowest y-values = high in the image/document
    bottomGroup = ySorted[abs(ySorted[-1,1] - ySorted[:,1]) < y_threshold] # highest y-values = low in the image/document
    
    '''Find top, bottom x extrema (leftmost and rightmost x within top group, bottom group)'''
    top_left, top_right = np.amin(topGroup[:,0]), np.amax(topGroup[:,0])
    bottom_left, bottom_right = np.amin(bottomGroup[:,0]), np.amax(bottomGroup[:,0])

    '''Perform in-group grouping (form top, bottom groups for left, right groups) [can/does this replace anything?]'''
    yGroup_xThresh = 5
    topLeft = topGroup[abs(top_left - topGroup[:,0]) < yGroup_xThresh] # leftTop = points in the leftGroup whose y distance from the left_yMin is < threshold
    topRight = topGroup[abs(top_right - topGroup[:,0]) < yGroup_xThresh]
    bottomLeft = bottomGroup[abs(bottom_left - bottomGroup[:,0]) < yGroup_xThresh]
    bottomRight = bottomGroup[abs(bottom_right - bottomGroup[:,0]) < yGroup_xThresh]

    '''Get fine-grained in-group leftmost and rightmost points'''
    topLeftRightmost, bottomLeftRightmost = np.amax(topLeft[:,0]), np.amax(bottomLeft[:,0])
    topRightLeftmost, bottomRightLeftmost = np.amin(topRight[:,0]), np.amin(bottomRight[:, 0]) 

    '''Get fine-grained in-group topmost and bottommost points'''
    topLeftBottommost, topRightBottommost = np.amax(topLeft[:,1]), np.amax(topRight[:,1]) # important one
    bottomLeftTopmost, bottomRightTopmost = np.amin(bottomLeft[:,1]), np.amin(bottomRight[:,1]) # important one 
    topLeftTopmost, topRightTopmost = np.amin(topLeft[:, 1]), np.amin(topRight[:,1]) # used for range checks
    bottomLeftBottommost, bottomRightBottommost = np.amax(bottomLeft[:,1]), np.amax(bottomRight[:,1]) # used for range checks 

    # <<----RANGES ----->> #
    '''Find x group y ranges'''
    top_xRange = top_right - top_left
    bottom_xRange = bottom_right - bottom_left
    '''Define in-group range check values'''
    topLeftYRange = topLeftBottommost - topLeftTopmost
    topRightYRange = topRightBottommost - topRightTopmost
    bottomLeftYRange = bottomLeftBottommost - bottomLeftTopmost
    bottomRightYRange = bottomRightBottommost - bottomRightTopmost


    # print("HERE'S topLeft, topRight, bottomLeft, bottomRight")
    # print(topLeft)
    # print()
    # print(topRight)
    # print()
    # print(bottomLeft)
    # print()
    # print(bottomRight)
    # print()


    # print("HERE ARE THE CORNER GROUPS in order (left, right, top, bottom)")
    # print(leftGroup)
    # print(rightGroup)
    # print(topGroup)
    # print(bottomGroup)
    # print()

    # print("left topmost is: " + str(left_top) + ". left bottommost is: " + str(left_bottom))
    # print("right topmost is: " + str(right_top) + ". right bottommost is: " + str(right_bottom))
    # print("top leftmost is: " + str(top_left) + ". top rightmost is: " + str(top_right))
    # print("bottom leftmost is: " + str(bottom_left) + ". bottom rightmost is: " + str(bottom_right))
    # print()

    # print("left y range: " + str(left_yRange))
    # print("right y range: " + str(right_yRange))
    # print("global y range: " + str(yRange))
    # print()

    # print("top x range: " + str(top_xRange))
    # print("bottom x range: " + str(bottom_xRange))
    # print("global x range: " + str(xRange))
    # print()

    '''<---------- CORNER ASSIGNMENT LOGIC ----------->'''
    ''' In the event that there is only 1 point within a group, the range will be 0'''

    '''<---------- LEFT CORNER ASSIGNMENT CHECKS ----------->'''
    '''For leftGroup, could be top left or bottom left corner'''
    tl = (leftTopRightmost, leftTopBottommost) 
    bl = (leftBottomRightmost, leftBottomTopmost) 

    # print("leftBottomXRange at " + str(leftBottomXRange))
    # print("leftTopXRange at " + str(leftBottomXRange))

    if leftTopXRange > 5:
        tl = (leftTopLeftmost, leftTopBottommost) 
    if leftBottomXRange > 5:
        # print("Fairly large leftBottomXRange at " + str(leftBottomXRange) + " ...resetting bl to leftBottomLeftMost")
        bl = (leftBottomLeftmost, leftBottomTopmost) 
    
    '''This check is here because we need a sufficiently large y-range to be confident that there is both a top and bottom corner in the group'''
    if left_yRange > 0.5 * yRange: # Document is likely axis-aligned. How to set these parameters? Does it matter?
        tl_set = True
        bl_set = True
        print("LEFT corner checks - SET BOTH tl AND bl (large left yRange)")
    else: # range must be small, so document is likely rotated
        if left_top <= midY: # topmost point in leftGroup is above the midline means set top left
            tl_set = True
            print("LEFT corner checks - SET JUST tl (small left yRange; left top is above the midline) ")
        else:
            bl_set = True
            print("LEFT corner checks - SET JUST bl (small left yRange; left top is below the midline) ")

    '''<---------- RIGHT CORNER ASSIGNMENT CHECKS ----------->'''
    tr = (rightTopLeftmost, rightTopBottommost)
    br = (rightBottomLeftmost, rightBottomTopmost)

    if rightTopXRange > 5:
        tr = (rightTopRightmost, rightTopBottommost) 
    if rightBottomXRange > 5:
        br = (rightBottomRightmost, rightBottomTopmost) 

    if right_yRange > 0.5 * yRange: # the range could not be this big unless the document is axis-aligned
        tr_set = True
        br_set = True
        print("RIGHT corner checks - SET BOTH tr AND br (large right yRange)")
    else:
        if right_top <= midY: # if the minY of the right group is above the document midline, it's top right
            tr_set = True
            print("RIGHT corner checks - SET JUST tr (small right yRange; right top is above the midline)")
        else: # otherwise, it must be below the document midline, and is therefore bottom right
            br_set = True
            print("RIGHT corner checks - SET JUST br (small right yRange; right top is below the midline)")

    '''<---------- TOP, BOTTOM CORNER CHECKS ----------->'''
    '''These are here to catch cases where corners were not set using the above logic
    This will occur (most notably) when the yRange within either the left or right group is small
    In those cases, only 1 corner in each of the left, right groups will be set, so the remaining corner has to be set here.'''

    tl_top = (topLeftRightmost, topLeftBottommost)
    tr_top = (topRightLeftmost, topRightBottommost)

    if topLeftYRange > 6:
        tl_top = (topLeftRightmost, topLeftTopmost)
    if topRightYRange > 6:
        tr_top = (topRightLeftmost, topRightTopmost)

    if tl[1] - tl_top[1] > 10: # If current tl is LOWER than tl_top, set tl to be tl_top
        print("RESET TL in top checks. Was " + str(tl) + " now " + str(tl_top))
        tl = tl_top
        tl_set = True
    if tr[1] - tr_top[1] > 10: # If current tr is LOWER than tr_top, set tr to be tr_top
        print("RESET TR in top checks. Was " + str(tr) + " now " + str(tr_top))
        tr = tr_top
        tr_set = True

    bl_bottom = (bottomLeftRightmost, bottomLeftTopmost)
    br_bottom = (bottomRightLeftmost, bottomRightTopmost)

    if bottomLeftYRange > 6:
        bl_bottom = (bottomLeftRightmost, bottomLeftBottommost)
    if bottomRightYRange > 6:
        br_bottom = (bottomRightLeftmost, bottomRightBottommost)
    
    if bl_bottom[1] - bl[1] > 10:
        print("RESET BL in bottom checks. Was " + str(bl) + " now " + str(bl_bottom))
        bl = bl_bottom
        bl_set = True
    if br_bottom[1] - br[1] > 10:
        print("RESET BR in bottom checks. Was " + str(br) + " now " + str(br_bottom))
        br = br_bottom
        br_set = True
    

    # '''<---------- TOP CORNER ASSIGNMENT CHECKS ----------->'''
    # if top_xRange > 0.7 * xRange: # Document is likely axis-aligned. How to set these parameters? Does it matter?
    #     '''I think this will fail with slight axis misalignment. Might want to grab points associated with top_xMin, rather than using with ymin'''
    #     print("Large top XRANGE")
    #     if not tl_set:
    #         tl = (topLeftRightmost, topLeftBottommost) # was (top_xMin, ymin)
    #         tl_set = True
    #         print("TOP corner checks - SET tl in 1st check")
    #     if not tr_set:
    #         tr = (topRightLeftmost, topRightBottommost) # was (top_xMax, ymin)
    #         tr_set = True
    #         print("TOP corner checks - SET tr in 1st check")
    # else: # range must be small, so document is likely rotated. Could be tl or tr here. 
    #     print("Small top XRANGE")
    #     if top_left <= midX: # difference between left_yMin and left_yMax should be negligible/shouldn't matter which. Maybe do average?
    #         if not tl_set:
    #             tl = (topLeftRightmost, topLeftBottommost) 
    #             tl_set = True
    #             print("TOP corner checks - SET tl in 2nd check (top xmin to left of midline)")
    #     else:
    #         if not tr_set:
    #             tr = (topRightLeftmost, topRightBottommost)
    #             tr_set = True
    #             print("TOP corner checks - SET tr in 2nd check (top xmax to right of midline)")
    
    # '''<---------- BOTTOM CORNER ASSIGNMENT CHECKS ----------->'''
    # if bottom_xRange > 0.5 * xRange:
    #     print("Large bottom XRANGE")
    #     if not bl_set:
    #         bl = (bottomLeftRightmost, bottomLeftTopmost) # was bottom_yMax, but they should be the same
    #         bl_set = True
    #         print("BOTTOM corner checks - SET bl in 1st check")
    #     if not br_set:
    #         br = (bottomRightLeftmost, bottomRightTopmost)
    #         br_set = True
    #         print("BOTTOM corner checks - SET br in 1st check")
    # else:
    #     print("Small bottom XRANGE")
    #     if bottom_left <= midX: # bottom xMin is to the left of the "document" midline, so is bottom left (bl) corner
    #         if not bl_set:
    #             bl = (bottomLeftRightmost, bottomLeftTopmost)
    #             bl_set = True    
    #             print("BOTTOM corner checks - SET bl in 2nd check (bottom xmin to left of midline)")
    #     else:
    #         if not br_set:
    #             br = (bottomRightLeftmost, bottomRightTopmost)
    #             br_set = True
    #             print("BOTTOM corner checks - SET br in 2nd check (bottom xmin to right of midline")



    if not tl_set:
        print("TOP LEFT NEVER SET")
    if not tr_set:
        print("TOP RIGHT NEVER SET")
    if not br_set:
        print("BOTTOM RIGHT NEVER SET")
    if not bl_set:
        print("BOTTOM LEFT NEVER SET")

    '''SUCCESS if all corners are set according to the above logic; failure otherwise'''
    if tl_set and tr_set and br_set and bl_set:
        success = True
    
    '''Form corners tuple and return results'''
    corners = (tl, tr, br, bl)
    corners = np.array(list(corners))
    
    return corners, success

    '''
    Get sorted X
    Get xMin group, get xMax group (points whose x dist from xMin, xMax is negligible)
        Sort these by y
            Get yMin group, yMax group PROBLEM is don't know whether xMin_pt should be the minX, minY value or minX, maxY value
                Compare assigned xMin to yMin, yMax? If identical, choose something else?
    '''



def corner_filtration(corners):
    xSorted = corners[np.argsort(corners[:, 0]), :] # this means sort by rows I think (row, col). Each row is a point. 
    ySorted = corners[np.argsort(corners[:, 1]), :]

    '''For a rotated document, this should work. For non-rotated, yMin and xMin pts could be the same'''
    xMin_pt = xSorted[0]
    xMax_pt = xSorted[-1]
    yMin_pt = ySorted[0]
    yMax_pt = ySorted[-1]

    '''Compute distance from suspected opposite corners. If negligible, get next Min_pt If none work, cornering failed'''
    '''X distance from yMin to yMax should be substantial (edge cases where rotated?)'''    
    for pt in ySorted[1:]: # loop through ySorted pts in ascending order (from smallest y value to biggest) (skip first point)
        if abs(yMin_pt[0] - yMax_pt[0]) < 10: # CHANGE to make some proportion of detected/cropped region
            '''If x distance between y extrema is insufficient, change either yMin or yMax (yMin I guess). Want opposite corners.'''
            yMin_pt = pt # set yMin_pt to be the next point in ySorted, continue to search
        else: # current distance is sufficient
            break

    for pt in xSorted[1:]: # loop over all points sorted by x value (looping in ascending order, lowest x to highest x)
        if abs(xMin_pt[1] - xMax_pt[1]) < 10: # CHANGE to make some proportion of detected/cropped region
            '''If y distance between x extrema is insufficient, change xMin. Want opposite corners.'''
            xMin_pt = pt # set xMin_pt to be the next point in xSorted, continue to search
        else: # current distance is sufficient
            break

    success = True
    '''If the distance between supposedly opposite corners is insufficient, did not successfully find 4 corner points'''
    if abs(xMin_pt[1] - xMax_pt[1]) < 15 or abs(yMin_pt[0] - yMax_pt[0]) < 15:
        print("FAILED corner filtration check; did not find realistic corners for document detection. ")
        success = False
    
    print(xMax_pt)
    xMin_pt, xMax_pt, yMin_pt, yMax_pt = tuple(xMin_pt), tuple(xMax_pt), tuple(yMin_pt), tuple(yMax_pt)
    corners = (xMin_pt, xMax_pt, yMin_pt, yMax_pt)
    return corners, success

def hough_corners(lines):
    corners = set()
    if not lines or len(lines) > 30:
        print("found either 0 or too many lines in standard hough")
        return corners
    else:
        corner_count = find_corners(lines)
        print("Here are the corner groups from find_corners")
        print(corner_count)
        # for group, point_set in corner_count.iteritems(): # doesn't work in Python 3
        for group, point_set in corner_count.items():
            points = np.vstack(list(point_set))
            x_average = int(np.mean(points[:, 0]))
            y_average = int(np.mean(points[:, 1]))
            corners.add((x_average,y_average))
    print("found " + str(len(corners)) + " corners in hough_corners after in-group average merging")
    return corners

def draw_corners(image, corners, color = (0,255,0)):
    drawing_image = image.copy()
    for corner in corners:
        cv2.circle(drawing_image,corner,2,color,-1)
        # cv2.circle(drawing_image,corner,8,color,-1)
    return drawing_image

def hough_cornering(image, orig, crop_box): # added original for plotting
    original = image.copy() # necessary? Need to walk through these methods and see what they change
    downsized, ratio = reshape.standard_resize(image, new_width = 100.0)
    # edged = edging.orig_page_edging(downsized) # bad 
    edged = edging.page_edging(downsized, thresh1=0, thresh2=160) # good
    # edged = edging.auto_edging(downsized, sigma=0.5) # not great
    lines = standard_hough_lines(edged)
    lined = draw_lines(downsized.copy(), lines)

    # functions.plot_image(edged)

    # corners = set()
    corners = np.zeros(shape=(4,2))
    if len(lines) < 70:
        # corners = hough_corners(lines) # COMMENTED OUT 4/18
        # print("FEWER than 30 lines...finding CORNERS")
        corners = all_corners(lines) # TRYING THIS 4/18
        # print("Here are the corners in hough cornering")
        # print(corners)
    else:
        print("TOO MANY HOUGH LINES AT " + str(len(lines)))

    success = False
    filtered_corners = np.zeros(shape=(4,2))
    upsized_filtered_corners = np.zeros(shape=(4,2))
    if len(corners) >= 4:
        # filtered_corners, success = corner_filtration(corner_array) # old method, didn't work.
        image_area = downsized.shape[0] * downsized.shape[1]
        filtered_corners, success = corner_filter(corners, image_area=image_area)
        # print("CORNERS after corner filtration")
        # print(filtered_corners)

        '''Compute bounding box size from corners. If too small, success = False. O
        OR add a check in the filtration method that ensures the range of all axes is reasonable
        Reason: came across a case where there were 4 points really close together that the logic interpreted as valid
        because it is using the global range defined in the method from the corners there '''

        # print("corners before upsizeing")
        # print(corners)
        # print(type(corners))
        upsized_corners = corners * ratio
        upsized_filtered_corners = filtered_corners * ratio

        upsized_corners = np.int0(upsized_corners)
        upsized_filtered_corners = np.int0(upsized_filtered_corners)
        
        xmin, xmax, ymin, ymax = boxing.max_points(crop_box)
        # print("UPSIZED CORNERS")
        # print(upsized_corners)
        # upsized_corners[:, 0] += xmin
        # upsized_corners[:, 1] += ymin

        # xmin, xmax, ymin, ymax = boxing.max_points(upsized_filtered_corners)
        # upsized_filtered_corners[:, 0] += xmin
        # upsized_filtered_corners[:, 1] += ymin

        upsized_filtered_corners[:,0] = np.add(upsized_filtered_corners[:,0], xmin, out=upsized_filtered_corners[:,0], casting="unsafe")
        upsized_filtered_corners[:,1] = np.add(upsized_filtered_corners[:,1], ymin, out=upsized_filtered_corners[:,1], casting="unsafe")

        # upsized_corners = tuple(map(tuple, upsized_corners))
        upsized_filtered_corners = tuple(map(tuple, upsized_filtered_corners))

    # cornered = draw_corners(downsized, corners)
    # cornered = draw_corners(original, upsized_corners)
    # cornered = draw_corners(orig, upsized_corners) # was here
    # if success:
        # cornered = draw_corners(cornered, filtered_corners, color=(255,0,0))
        # cornered = draw_corners(cornered, upsized_filtered_corners, color=(255,0,0)) # was here
        # warped = reshape.perspective_transform(image=original, points=filtered_corners, ratio=ratio)
        # warped = reshape.perspective_transform(image=orig, points=upsized_filtered_corners)
        # threshed = coloring.thresholding(warped)
        # functions.plot_images([image, lined, cornered, threshed], titles=['Original', 'Hough lines', 'Corners', 'Transform'])
        # functions.plot_images([orig, image, edged, lined, cornered, threshed], titles=['Original', 'Cropped', 'Canny', 'Hough lines', 'Corners', 'Transform']) # was here
        # functions.plot_images([original, edged, lined, cornered, threshed], titles=['Original', 'edged', 'Hough lines', 'Corners', 'Transform'])
    # else:
        # functions.plot_images([image, edged, lined, cornered], titles=['image', 'edged', 'lined', 'cornered'])    # was here
        # print('hello dolly')
    

    '''This works when 4 corners are passed in. Need to figure out how to filter down to 4 corners or skip it if no corners found.'''
    return upsized_filtered_corners, success
    # functions.plot_images([image, edged, lined, cornered], titles=['image', 'edged', 'lined', 'cornered'])
    # functions.plot_images([image, lined, cornered, warped])


def hough_cornering_draw(image, orig, crop_box): # added original for plotting
    original = image.copy() # necessary? Need to walk through these methods and see what they change
    downsized, ratio = reshape.standard_resize(image, new_width = 100.0)
    
    # edged = edging.page_edging(downsized, thresh1=75, thresh2=200) # good
    edged = edging.new_page_edging(downsized, thresh1 = 0, thresh2 = 160)
    # edged = edging.auto_edging(downsized, sigma=0.33) # not great
    # edged = functions.orig_page_edging(downsized)
    # edged = functions.page_edging(downsized, thresh1=0, thresh2=120)
    # edged = functions.page_edging(original, thresh1=0, thresh2=120)
    # edged = functions.orig_page_edging(original)
    
    lines = standard_hough_lines(edged)
    lined = draw_lines(downsized.copy(), lines)

    corners = np.zeros(shape=(4,2))
    if len(lines) < 70:
        # corners = hough_corners(lines) # COMMENTED OUT 4/18
        # print("FEWER than 30 lines...finding CORNERS")
        corners = all_corners(lines) # TRYING THIS 4/18
        # print("Here are the corners in hough cornering")
        # print(corners)
    else:
        print("TOO MANY HOUGH LINES AT " + str(len(lines)))

    success = False
    filtered_corners = np.zeros(shape=(4,2))
    upsized_corners = np.zeros(shape=(4,2))
    upsized_filtered_corners = np.zeros(shape=(4,2))
    if len(corners) >= 4:
        # filtered_corners, success = corner_filtration(corner_array) # old method, didn't work.
        image_area = downsized.shape[0] * downsized.shape[1]
        filtered_corners, success = corner_filter(corners, image_area=image_area)
        # print("CORNERS after corner filtration")
        # print(filtered_corners)

        '''Compute bounding box size from corners. If too small, success = False. O
        OR add a check in the filtration method that ensures the range of all axes is reasonable
        Reason: came across a case where there were 4 points really close together that the logic interpreted as valid
        because it is using the global range defined in the method from the corners there '''

        # print("corners before upsizeing")
        # print(corners)
        # print(type(corners))
        upsized_corners = corners * ratio
        upsized_filtered_corners = filtered_corners * ratio

        upsized_corners = np.int0(upsized_corners)
        upsized_filtered_corners = np.int0(upsized_filtered_corners)
        
        xmin, xmax, ymin, ymax = boxing.max_points(crop_box)
        # print("UPSIZED CORNERS")
        # print(upsized_corners)
        # upsized_corners[:, 0] += xmin
        # upsized_corners[:, 1] += ymin

        # xmin, xmax, ymin, ymax = boxing.max_points(upsized_filtered_corners)
        # upsized_filtered_corners[:, 0] += xmin
        # upsized_filtered_corners[:, 1] += ymin

        upsized_corners[:,0] = np.add(upsized_corners[:,0], xmin, out=upsized_corners[:,0], casting="unsafe")
        upsized_corners[:,1] = np.add(upsized_corners[:,1], ymin, out=upsized_corners[:,1], casting="unsafe")

        upsized_filtered_corners[:,0] = np.add(upsized_filtered_corners[:,0], xmin, out=upsized_filtered_corners[:,0], casting="unsafe")
        upsized_filtered_corners[:,1] = np.add(upsized_filtered_corners[:,1], ymin, out=upsized_filtered_corners[:,1], casting="unsafe")

        # upsized_corners = tuple(map(tuple, upsized_corners))
        # upsized_filtered_corners = tuple(map(tuple, upsized_filtered_corners))
    corners = np.int32(corners)
    corners = tuple(map(tuple, corners))
    filtered_corners = np.int32(filtered_corners)
    filtered_corners = tuple(map(tuple, filtered_corners))

    upsized_corners = np.int32(upsized_corners)
    upsized_filtered_corners = np.int32(upsized_filtered_corners)
    upsized_corners = tuple(map(tuple, upsized_corners))
    upsized_filtered_corners = tuple(map(tuple, upsized_filtered_corners))
    cornered = draw_corners(downsized, corners)
    # cornered = draw_corners(original, upsized_corners)
    # cornered = draw_corners(orig, upsized_corners) # was here
    if success:
        cornered = draw_corners(cornered, filtered_corners, color=(255,0,0))
        # cornered = draw_corners(cornered, upsized_filtered_corners, color=(255,0,0)) # was here
        # warped = reshape.perspective_transform(image=original, points=filtered_corners, ratio=ratio)
        warped = reshape.perspective_transform(image=orig, points=upsized_filtered_corners)
        threshed = coloring.thresholding(warped)
        # functions.plot_images([orig, lined, cornered, threshed], titles=['Original', 'Hough lines', 'Corners', 'Transform'])
        functions.plot_images([orig, image, edged, lined, cornered, threshed], titles=['Original', 'Cropped', 'Canny', 'Hough lines', 'Corners', 'Transform']) # was here
        # functions.plot_images([original, edged, lined, cornered, threshed], titles=['Original', 'edged', 'Hough lines', 'Corners', 'Transform'])
    else:
        # functions.plot_images([orig, image, edged, lined], titles=['Original', 'Cropped', 'Edged', 'Lined'])    # was here
        functions.plot_images([orig, downsized, edged, lined], titles=['Original', 'Cropped', 'Edged', 'Lined'])    # was here
        # print('hello dolly')
    

    '''This works when 4 corners are passed in. Need to figure out how to filter down to 4 corners or skip it if no corners found.'''
    
    # functions.plot_images([image, edged, lined, cornered], titles=['image', 'edged', 'lined', 'cornered'])
    # functions.plot_images([image, lined, cornered, warped])
    return upsized_filtered_corners, success



def hough_video(frame):
    orig = frame.copy()
    downsized, ratio = reshape.standard_resize(image=frame, new_width = 100.0)
    edged = edging.page_edging(downsized, thresh1=0, thresh2=160) # good
    # lines = standard_hough_lines(edged)
    # lined = draw_lines(downsized.copy(), lines, ratio)
    # lined = draw_lines(drawing_image=orig, lines = lines)
    # return lined
    return edged

def hough_detection(image):
    # original = image.copy() # necessary? Need to walk through these methods and see what they change
    downsized, ratio = reshape.standard_resize(image, new_width = 100.0)
    edged = edging.page_edging(downsized, thresh1=0, thresh2=160) # good
    lines = standard_hough_lines(edged)

    # corners = np.zeros(shape=(4,2))
    # if len(lines) < 70:
    corners = all_corners(lines) # TRYING THIS 4/18

    success = False
    # if len(corners) >= 4 and np.linalg.norm(corners) > 0:
    if len(corners) >= 4:
        downsized_area = downsized.shape[0] * downsized.shape[1]
        filtered_corners, success = corner_filter(corners, image_area=downsized_area)
        corners = np.array(list(filtered_corners))
        print("original corners: " + str(corners))
        corners = corners * ratio
    
    print("final corners: " + str(corners))
    return corners, success



if __name__ == '__main__':
    # files = utility.image_paths() 
    files = ["data/pics/forms/sample4_3.jpg", "data/pics/forms/sample4_4.jpg", "data/pics/invoice2/sample1.JPG", "data/pics/invoice2/sample1_2.JPG"]
    files = ["data/pics/forms/sample5.jpg", "data/pics/forms/sample2.jpg", "data/pics/forms/sample3.jpg", "data/pics/forms/sample4_4.jpg", "data/pics/forms/sample9.jpg", "data/pics/forms/sample11.jpg", "data/pics/forms/sample8.jpg", "data/pics/forms/sample12.jpg"]
    files = ["data/pics/forms/sample5.jpg", "data/pics/forms/sample2.jpg", "data/pics/forms/sample4_4.jpg", "data/pics/forms/sample11.jpg", "data/pics/forms/sample12.jpg", "data/pics/forms/sample10.jpg"]
    # files = ["data/pics/demo/IMAG0603.jpg", "data/pics/demo/IMAG0604.jpg", "data/pics/demo/IMAG0605.jpg", "data/pics/demo/IMAG0606.jpg", "data/pics/demo/IMAG0607.jpg", "data/pics/demo/IMAG0608.jpg", "data/pics/demo/IMAG0611.jpg", "data/pics/demo/IMAG0612.jpg"]
    # files = ["data/pics/forms/E-Ticketing.png", "data/pics/indoor_720.jpg", "data/pics/forms/sample4_2.jpg"]

    # files = utility.filenames_at_path("/media/thor/LEXAR/sampleDataset/input_sample", ".jpg") # hough threshold of 30 is best here
    # files = utility.filenames_at_path("/home/thor/code/sr_project/data/pics/forms", ".jpg")
    files = ["data/pics/forms/sample2.jpg"]

    originals, images = image_reading(files[:20])

    for image in images:
        hough_cornering(image)

    # functions.plot_images(originals, files)

    downsized = functions.process_several(originals, function = reshape.standard_resize, return_ratio = False)
    downs = list(downsized)
    # # functions.plot_images(downsized)
    # edged = functions.process_several(downsized, function = functions.page_edging) # was this
    edged = functions.process_several(downsized, function = functions.orig_page_edging)
    # processed = functions.process_several(edged, function = functions.closed_inversion)

    result = functions.draw_several(edged, downs, function = standard_hough)  
    # functions.plot_images(originals + edged + result)
    # functions.plot_images([originals, edged, result])




    # time_downsized_hough = timing(function = downsized_hough, originals = originals, images = images)
    # print "time_downsized_hough took " + str(time_downsized_hough) + " seconds"
    # time_hough_downsized = timing(function = hough_downsized, originals = originals, images = images)
    # print "time_hough_downsized took " + str(time_hough_downsized) 
    # print "downsized hough took " + str(time_downsized_hough/time_hough_downsized * 100) + " % of hough downsized"

 
