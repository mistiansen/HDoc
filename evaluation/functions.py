import cv2
import sys
import numpy as np
import argparse
import imutils
import math
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive
 

# <---- IMAGE COLOR, SMOOTHING, AND CANNY EDGE DETECTION PRE-PROCESSING -----> #

def process_several(images, function, **kwargs):
    results = []
    for image in images:
        result = function(image, **kwargs)
        results.append(result)
    # plot_images(results)
    return results 

def draw_several(images, drawing_images, function, **args): # this really only gets used with HoughLines, where we have different images for processing vs. drawing on 
    if not len(drawing_images) == len(images):
        # print("In display_several, number of images and images to draw on are not the same")
        return 
    else:
        drawn_images = []
        for i, image in enumerate(images):
            drawn_image = function(image, drawing_images[i], **args)
            drawn_images.append(drawn_image)
    return drawn_images


def draw_several2(images, drawing_images, function, **args): # this really only gets used with HoughLines, where we have different images for processing vs. drawing on 
    if not len(drawing_images) == len(images):
        # print( "In display_several, number of images and images to draw on are not the same"
        return 
    else:
        drawn_images = []
        for i, image in enumerate(images):
            _, drawn_image = function(image, drawing_images[i], **args)
            drawn_images.append(drawn_image)
    return drawn_images

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

def shadow_removal(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV) # convert BGR to YUV
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) # perform histogram equalization of y channel
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # convert back to BGR
    # cv2.imshow('Color input image', image)
    # cv2.waitKey(0)
    # cv2.imshow('Histogram equalized', img_output)
    # cv2.imshow('Histogram equalized', img_yuv)
    # cv2.waitKey(0)
    return img_output

def gamma_correction(image, correction):
    image = image/255.0
    image = cv2.pow(image, correction)
    return np.uint8(image*255)

def colorOps(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv2.bilateralFilter
    # blurred = cv2.copyMakeBorder(blurred, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=[0, 0, 0])		 # added this 11/2/17. Trying to work with document occlusion. 
    edged = cv2.Canny(blurred, threshold1 = 75, threshold2 = 200) # was 0, 50...not sure what these numbers mean
    # edged_invert = edged.copy()
    # cv2.bitwise_not(edged, edged_invert)
    # plot_images([edged, edged_invert])
    return edged

def closed_inversion(image):
    closedEdges = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel = np.ones((5, 11))) # this potentially helps close Canny edges that are close but not quite touching
    closedInvert = cv2.bitwise_not(src = closedEdges.copy())
    
    kernel = np.ones((3,3),np.uint8) # original was 15x15. 5x5 is working well right now. 
    erosion = cv2.erode(closedInvert, kernel, iterations=1) # THIS LOOKS GOOD ON ALL but s4!! It erodes too much though. Losing some of document.
    # erosion = cv2.erode(closedEdges, kernel, iterations=1) # Attempting to return a non-inverted version for use with HoughLinesP in new.py -> DOESN'T WORK
    dilation = cv2.dilate(erosion,kernel,iterations = 1)

    reInvert = cv2.bitwise_not(src = erosion) # attempting to reInvert the eroded 

    # plot_images([closedEdges, closedInvert, erosion, dilation, reInvert]) # for testing 
    # return closedInvert
    return erosion # THIS WORKS WELL FOR PRETTY MUCH ALL DOCUMENTS SO FAR (except sample4)
    # return reInvert # return this for standard hough I guess, since I think it looks for white

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
    print("print(ing target: " + str(target))
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


def all_boxes(image):
    contours = fetch_largest_contours(image)
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
        print("Got a quadrant/corner point")
        return True
    else:
        return False

# def alternateRectMethod(image):
def box_generation(image):
    contours = fetch_largest_contours(image)
    # detected = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB) # this is just so can see the green bounding box
    image_width, image_height = image.shape[1], image.shape[0]
    image_area = image_width * image_height
    boxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        rect_angle = rect[2]
        # if rect_center_in_quadrant(rect=rect, image_width=image_width, image_height=image_height) or  45 < abs(rect_angle) < 90:
        if rect_center_in_quadrant(rect=rect, image_width=image_width, image_height=image_height):
            continue
        print("Passed rect quadrant check")
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print("IN BOX GENERATION")
        print(type(box))
        print(box.shape)
        print(box)
        box_area = cv2.contourArea(box)
        if (box_area > (0.88 * image_area) or box_area < 0.10 * image_area or box_in_image_half(box, image_width, image_height)):
        # if (box_area > (0.90 * image_area) or box_area < 0.10 * image_area):
            # print( "box rejected in alternateRectMethod. Area was: " + str(box_area)
            continue
        # elif (box_area > (0.75 * image_area) and (abs(rect[2]) == 0 or abs(rect[2]) == 90)):
            # print( "WNTING TO DISCARD LARGE, ALIGNED BOX " + str(box_area/image_area * 100) + "% of image area"
        else:
            # print( "rect is " + str(rect)
            # print( "box area is " + str(box_area/image_area * 100) + "% of image area"
            # w, h = rect[1][:2] 
            # w = w - 8 # this is to recover some loss from the morphological operations
            # h = h - 8 # this is to recover some loss from the morphological operations
            # new_rect = (rect[0], (w,h), rect[2])
            # box = cv2.boxPoints(new_rect)
            # box = np.int0(box)
            print("PRINTING RECT IN FUNCTIONS.BOX_GENERATION")
            print(rect)
            print("PRINTING BOX IN FUNCTIONS.BOX_GENERATION")
            print(box)
            boxes.append(box)
    return boxes

# if a box is axis-aligned and too close to the image's edges, it is most likely a false positive.
def box_filtration(boxes, image_width, image_height):
    filtered = []
    for box in boxes:
        rect = cv2.minAreaRect(box)
        angle = rect[2]
        print("angle in box_filtration " + str(abs(angle)))
        if abs(angle) == 0 or abs(angle) == 90:
            xmin, xmax, ymin, ymax = box_extrema(box)
            # if -2 < xmin < 2 and (-2 < ymin < 2 or image_height - 0.02 * image_height <= ymax <= 1.02 * image_height):
            if -2 < xmin < 2 and (-2 < ymin < 2 or image_height - 2 <= ymax <= image_height + 2):
                print("FILTERED A BOX FROM THE TOP")
                continue
            # elif image_width - (0.02 * image_width) <= xmax <= 1.02 * image_width and ( -2 <= ymin <= 2 or image_height - (0.02 * image_height) <= ymax <= (1.02 * image_height)):
            elif image_width - 2 <= xmax <= image_width + 2 and ( -2 <= ymin <= 2 or image_height - 2 <= ymax <= image_height + 2):
                print("FILTERED A BOX FROM THE BOTTOM")
                continue
            else:
                filtered.append(box)
        else:
            filtered.append(box)
    print("filtered boxes " + str(boxes))
    return filtered

    

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
        print("merging points: " + str(points))
        xSorted = points[np.argsort(points[:, 0]), :] # this means sort by rows I think (row, col). Each row is a point. 
        ySorted = points[np.argsort(points[:, 1]), :]
        xMin = xSorted[0][0]
        xMax = xSorted[rows - 1][0]
        # xMax = xSorted[-1][0]
        yMin = ySorted[0][1]
        yMax = ySorted[rows - 1][1]
        # yMax = ySorted[-1][1]
        merged = np.array([[xMin, yMax], [xMin, yMin], [xMax, yMin], [xMax, yMax]]) # looks like OpenCV orders bottom left, top left, top right, bottom right
    print("merged is " + str(merged))
    return merged    


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

def box_crop(image, box):
    xmin, xmax, ymin, ymax = max_points(box)
    return image[ymin:ymax+1, xmin:xmax+1].copy()



def perspective_transform(image, points, ratio = None): 
    # print( "print(ing points in finalize " + str(points)
    height, width = image.shape[0], image.shape[1]
    if ratio:
        points = points.reshape(4,2) * ratio
    else:
        points = points.reshape(4,2)
    # points = imutils.boundary_coords(points, width, height)
    warped = imutils.four_point_transform(image = image.copy(), pts = points) # for use with minAreaRect resul
    return warped

def thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binarized = threshold_adaptive(gray, 251, offset = 10) # this is the original but it uses skimage
    # final = binarized.astype("uint8") * 255 # this is the original from skimage
    # gauss_thresh = cv2.adaptiveThreshold(src = gray, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #     thresholdType = cv2.THRESH_BINARY, blockSize = 251, C = 10) # blocksize was 11. This is good for a page
    gauss_thresh = cv2.adaptiveThreshold(src = gray, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        thresholdType = cv2.THRESH_BINARY, blockSize = 11, C = 10) # blocksize was 11. This is good for a page
    return gauss_thresh



def standard_resize(image, new_width = 100.0, return_ratio = True):
        # <---- RESIZING -----> #
    height, width = image.shape[:2]
    # print( "initial height, width is : " + str(height) + " " + str(width)
    # new_width = 100.0
    print("std_resize width, height: " + str(width) + ", " + str(height))
    scaling_factor = new_width/width
    ratio = 1/scaling_factor
    resized = imutils.resize_new(image, scaling_factor = scaling_factor)
    # print( "resized height, width is : " + str(resized.shape[0]) + " " + str(resized.shape[1]) + " with area " + str(resized.shape[0]*resized.shape[1])
    if return_ratio:
        return resized, ratio
    else:
        return resized
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


def dilation_canny(image):
    kernel = np.ones((5,5),np.uint8) # original was 15x15. 5x5 is working well right now. 
    dilation = cv2.dilate(image,kernel,iterations = 1)
    resized_image = standard_resize(image)
    resized_dilation = standard_resize(dilation)
    edged_resized = colorOps(resized_image)
    edged_dilation = colorOps(resized_dilation)
    plot_images([dilation, edged_resized, edged_dilation])
    

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



# <-----  UTILITY FUNCTIONS  ------> #

def plot_images(images, titles = None):
    if len(images) == 1:
        if titles:
            plot_image(images[0], titles[0])
        else:
            plot_image(images[0])
    elif len(images) < 5:
        for i in range(len(images)):
            plt.subplot(1, len(images), i+1), plt.imshow(images[i], 'gray')
            if titles:
                plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()   
    else:
        for i in range(len(images)):
            plt.subplot(2, int(round(len(images)/2.0)), i+1),plt.imshow(images[i],'gray') # nrows, ncols, plot_index. Arg names don't work for some reason.
            if titles:
                plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()   

def plot_image(image, title = None):
    plt.imshow(image, 'gray')
    if title:
        plt.title(title)
    plt.xticks([]),plt.yticks([])
    plt.show()

def save_images(names, images):
    for i in range(len(images)):
        cv2.imwrite(names[i], images[i])


if __name__ == '__main__':
    image = cv2.imread(sys.argv[1])



