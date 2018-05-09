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


def draw_lines(drawing_image, lines, ratio = None): # modifies the passed in image
    for line in lines:
        pt1, pt2 = line
        if ratio:
            pt1, pt2 = tuple(int(round(coord * ratio)) for coord in pt1), tuple(int(round(coord * ratio)) for coord in pt2)
        cv2.line(img = drawing_image, pt1 = pt1, pt2 = pt2, color = (255,0,0), thickness = 1)
    return drawing_image

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
        return corners, success
        # return (tl, tr, br, bl), success

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

    print("HERE'S leftTop, leftBottom, rightTop, rightBottom")
    print(leftTop)
    print()
    print(leftBottom)
    print()
    print(rightTop)
    print()
    print(rightBottom)
    print()

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


    print("HERE'S topLeft, topRight, bottomLeft, bottomRight")
    print(topLeft)
    print()
    print(topRight)
    print()
    print(bottomLeft)
    print()
    print(bottomRight)
    print()


    print("HERE ARE THE CORNER GROUPS in order (left, right, top, bottom)")
    print(leftGroup)
    print(rightGroup)
    print(topGroup)
    print(bottomGroup)
    print()

    print("left topmost is: " + str(left_top) + ". left bottommost is: " + str(left_bottom))
    print("right topmost is: " + str(right_top) + ". right bottommost is: " + str(right_bottom))
    print("top leftmost is: " + str(top_left) + ". top rightmost is: " + str(top_right))
    print("bottom leftmost is: " + str(bottom_left) + ". bottom rightmost is: " + str(bottom_right))
    print()

    print("left y range: " + str(left_yRange))
    print("right y range: " + str(right_yRange))
    print("global y range: " + str(yRange))
    print()

    print("top x range: " + str(top_xRange))
    print("bottom x range: " + str(bottom_xRange))
    print("global x range: " + str(xRange))
    print()

    '''<---------- CORNER ASSIGNMENT LOGIC ----------->'''
    ''' In the event that there is only 1 point within a group, the range will be 0'''

    '''<---------- LEFT CORNER ASSIGNMENT CHECKS ----------->'''
    '''For leftGroup, could be top left or bottom left corner'''
    tl = (leftTopRightmost, leftTopBottommost) 
    bl = (leftBottomRightmost, leftBottomTopmost) 

    print("leftBottomXRange at " + str(leftBottomXRange))
    print("leftTopXRange at " + str(leftBottomXRange))

    if leftTopXRange > 5:
        tl = (leftTopLeftmost, leftTopBottommost) 
    if leftBottomXRange > 5:
        print("Fairly large leftBottomXRange at " + str(leftBottomXRange) + " ...resetting bl to leftBottomLeftMost")
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


def hough_cornering(image, orig): # added original for plotting
    original = image.copy() # necessary? Need to walk through these methods and see what they change
    downsized, ratio = reshape.standard_resize(image, new_width = 100.0)
    # edged = edging.orig_page_edging(downsized) # bad 
    edged = edging.page_edging(downsized, thresh1=0, thresh2=160) # good
    # edged = edging.auto_edging(downsized, sigma=0.5) # not great
    lines = standard_hough_lines(edged)
    lined = draw_lines(downsized.copy(), lines)

    # functions.plot_image(edged)

    corners = set()
    if len(lines) < 70:
        # corners = hough_corners(lines) # COMMENTED OUT 4/18
        print("FEWER than 30 lines...finding CORNERS")
        corners = all_corners(lines) # TRYING THIS 4/18
        print("Here are the corners in hough cornering")
        print(corners)
    else:
        print("TOO MANY HOUGH LINES AT " + str(len(lines)))

    success = False
    if len(corners) >= 4:
        # filtered_corners, success = corner_filtration(corner_array) # old method, didn't work.
        image_area = downsized.shape[0] * downsized.shape[1]
        filtered_corners, success = corner_filter(corners, image_area=image_area)
        print("CORNERS after corner filtration")
        print(filtered_corners)

    '''Compute bounding box size from corners. If too small, success = False. O
        OR add a check in the filtration method that ensures the range of all axes is reasonable
        Reason: came across a case where there were 4 points really close together that the logic interpreted as valid
        because it is using the global range defined in the method from the corners there '''

    upsized_corners = corners * ratio
    upsized_filtered_corners = filtered_corners * ratio

    upsized_corners = np.int0(upsized_corners)
    upsized_filtered_corners = np.int0(upsized_filtered_corners)

    upsized_corners = tuple(map(tuple, upsized_corners))
    upsized_filtered_corners = tuple(map(tuple, upsized_filtered_corners))

    # cornered = draw_corners(downsized, corners)
    cornered = draw_corners(original, upsized_corners)
    if success:
        # cornered = draw_corners(cornered, filtered_corners, color=(255,0,0))
        cornered = draw_corners(cornered, upsized_filtered_corners, color=(255,0,0))
        warped = reshape.perspective_transform(image=original, points=filtered_corners, ratio=ratio)
        threshed = coloring.thresholding(warped)
        # functions.plot_images([image, lined, cornered, threshed], titles=['Original', 'Hough lines', 'Corners', 'Transform'])
        functions.plot_images([orig, image, edged, lined, cornered, threshed], titles=['Original', 'Cropped', 'Canny', 'Hough lines', 'Corners', 'Transform'])
        # functions.plot_images([original, edged, lined, cornered, threshed], titles=['Original', 'edged', 'Hough lines', 'Corners', 'Transform'])
    else:
        functions.plot_images([image, edged, lined, cornered], titles=['image', 'edged', 'lined', 'cornered'])    

def hough_corners(image, orig, crop_box): # added original for plotting
    original = image.copy() # necessary? Need to walk through these methods and see what they change
    downsized, ratio = reshape.standard_resize(image, new_width = 100.0)
    edged = edging.page_edging(downsized, thresh1=0, thresh2=160) # good
    lines = standard_hough_lines(edged)
    # lined = draw_lines(downsized.copy(), lines)

    corners = set()
    if len(lines) < 70:
        corners = all_corners(lines) # TRYING THIS 4/18
    else:
        print("TOO MANY HOUGH LINES AT " + str(len(lines)))

    success = False
    if len(corners) >= 4:
        image_area = downsized.shape[0] * downsized.shape[1]
        filtered_corners, success = corner_filter(corners, image_area=image_area)

    # upsized_corners = corners * ratio
    upsized_filtered_corners = filtered_corners * ratio

    # upsized_corners = np.int0(upsized_corners)
    upsized_filtered_corners = np.int0(upsized_filtered_corners)
    
    xmin, xmax, ymin, ymax = boxing.max_points(crop_box)
    # upsized_corners[:, 0] += xmin
    # upsized_corners[:, 1] += ymin
    upsized_filtered_corners[:, 0] += xmin
    upsized_filtered_corners[:, 1] += ymin

    # upsized_corners = tuple(map(tuple, upsized_corners))
    upsized_filtered_corners = tuple(map(tuple, upsized_filtered_corners))

    # cornered = draw_corners(orig, upsized_corners)
    # if success:
        # cornered = draw_corners(cornered, upsized_filtered_corners, color=(255,0,0))
        # warped = reshape.perspective_transform(image=orig, points=upsized_filtered_corners)
        # threshed = coloring.thresholding(warped)
        # functions.plot_images([orig, image, edged, lined, cornered, threshed], titles=['Original', 'Cropped', 'Canny', 'Hough lines', 'Corners', 'Transform'])
    '''This works when 4 corners are passed in. Need to figure out how to filter down to 4 corners or skip it if no corners found.'''
    return upsized_filtered_corners

def hough_detection(image):
    original = image.copy() # necessary? Need to walk through these methods and see what they change
    downsized, ratio = reshape.standard_resize(image, new_width = 100.0)
    edged = edging.page_edging(downsized, thresh1=0, thresh2=160) # good
    lines = standard_hough_lines(edged)

    corners = set()
    if len(lines) < 70:
        corners = all_corners(lines) # TRYING THIS 4/18
    else:
        print("TOO MANY HOUGH LINES AT " + str(len(lines)))

    success = False
    if len(corners) >= 4:
        downsized_area = downsized.shape[0] * downsized.shape[1]
        filtered_corners, success = corner_filter(corners, image_area=downsized_area)

    upsized_filtered_corners = filtered_corners * ratio
    upsized_filtered_corners = np.int0(upsized_filtered_corners)
    
    # xmin, xmax, ymin, ymax = boxing.max_points(crop_box)
    # upsized_filtered_corners[:, 0] += xmin
    # upsized_filtered_corners[:, 1] += ymin

    # upsized_filtered_corners = tuple(map(tuple, upsized_filtered_corners))
    return upsized_filtered_corners, success

    

def draw_corners(image, corners, color = (0,255,0)):
    drawing_image = image.copy()
    for corner in corners:
        cv2.circle(drawing_image,corner,1,color,-1)
    return drawing_image


# def hough_detection(image):
#     # original = image.copy() # necessary? Need to walk through these methods and see what they change
#     downsized, ratio = reshape.standard_resize(image, new_width = 100.0)
#     edged = edging.page_edging(downsized, thresh1=0, thresh2=160) # good
#     lines = standard_hough_lines(edged)

#     # corners = np.zeros(shape=(4,2))
#     # if len(lines) < 70:
#     corners = all_corners(lines) # TRYING THIS 4/18

#     success = False
#     # if len(corners) >= 4 and np.linalg.norm(corners) > 0:
#     if len(corners) >= 4:
#         downsized_area = downsized.shape[0] * downsized.shape[1]
#         filtered_corners, success = corner_filter(corners, image_area=downsized_area)
#         corners = np.array(list(filtered_corners))
#         print("original corners: " + str(corners))
#         corners = corners * ratio
    
#     print("final corners: " + str(corners))
#     return corners, success