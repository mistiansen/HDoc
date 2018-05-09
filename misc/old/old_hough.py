import cv2
import math
import numpy as np
import sys
from matplotlib import pyplot as plt
import imutils
import functions
import text_regions
import decimal
import utility

def get_new(old):
    new = np.ones(old.shape, np.uint8)
    cv2.bitwise_not(new,new)
    return new

def edged_gradient(image, kernel_size = (15,15)):
    kernel = np.ones(kernel_size,np.uint8)
    # dilation = cv2.dilate(img,kernel,iterations = 1) # why was this working? The variable is called 'image' not 'img'. Is it using the wrong non-local variable??
    dilation = cv2.dilate(image, kernel, iterations = 1) # iterations was 1
    dilated_edged = cv2.Canny(dilation, 0, CANNY, apertureSize = 3) # done in this order because the dilation first gets rid of the text, then do edge detection
    gradient = cv2.morphologyEx(dilated_edged, cv2.MORPH_GRADIENT, kernel)  
    return gradient
    # return dilated_edged

def new_technique(image):
    original = image.copy()
    outlined = edged_gradient(image)

def computeIntersect(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = a[2]
    y2 = a[3]
    x3 = b[0]
    y3 = b[1]
    x4 = b[2]
    y4 = b[3]
    d = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    
    if d:
        d = float(d)
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
        return int(x), int(y)
    else:
        return -1, -1

def find_corners(lines): # lines is output of houghlinesP
    corners = []
    for i, lineI in enumerate(lines): # enumerate just creates a counter that you can use along with the variable in a for loop
        for lineJ in lines[i+1:]:
            x, y = computeIntersect(lineI, lineJ)
            if x >=0 and y >= 0:
                corners.append((x, y))
    return corners
    # corner_draw = image.copy()    
    # for x1,y1,x2,y2 in lines:
    #     cv2.line(corner_draw,(x1,y1),(x2,y2),(0,255,0),2) # put this code wherever want to draw corners


def standard_hough(image, drawing_image):
    lines = []
    count = dict()
    # hough_lines = cv2.HoughLines(image, rho = 1, theta = 3.14/180, threshold = 250)  # this works pretty well for normal "edged" input (assuming no illustrations in image)
    # hough_lines = cv2.HoughLines(image, rho = 1, theta = 3.14/180, threshold = 950)  # need much higher threshold for "dilated" input, because is so clear
    hough_lines = cv2.HoughLines(image, rho = 1, theta = 3.14/180, threshold = 30) # when using downsized image for corner finding, need very low threshold
    if hough_lines is None:
        print "NO LINES FOUND"
    else:    
        print "number of lines found " + str(hough_lines.shape)
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
                line = ((x1,y1), (x2,y2))
                lines.append(line)
    print count

    for c in sorted(count, key=count.get, reverse=True):
        print c, count[c]

    theta = max(count, key=count.get)
    standard_hough_rotation(theta, drawing_image)

    return lines, drawing_image

def standard_hough_rotation(theta, drawing_image):
    # rotation_angle = ((math.cos(theta) * 180) / math.pi) * -1
    # rotation_angle = (90 - math.degrees(theta)) * -1
    rotation_angle = (90 - math.degrees(theta)) * -1
    rotated_image = functions.rotate(drawing_image, rotation_angle)
    print "most common theta is: " + str(theta)
    print "rotation angle is: " + str(rotation_angle)
    functions.plot_images([image, drawing_image, rotated_image], ["input image", "Std. Hough", "Rotated"])



def prob_hough(image, drawing_image):
    hough_lines = cv2.HoughLinesP(image, rho = 1, theta = 3.14/180, threshold = 250, minLineLength = image.shape[:2][1]/8, maxLineGap = image.shape[:2][1]/50) 
    # hough_lines = cv2.HoughLinesP(image, rho = 1, theta = 3.14/180, threshold = 25, minLineLength = image.shape[:2][1]/10, maxLineGap = image.shape[:2][1]/50)  # working fairly well
    # lines = cv2.HoughLinesP(edges, 1,  3.14/180, HOUGH)
    # lines = cv2.HoughLinesP(edged_grad, rho = 1, theta = 3.14/180, threshold = 100, minLineLength = 150, maxLineGap = HOUGH)
    # lines = cv2.HoughLinesP(edged, rho = 1, theta = 3.14/180, threshold = 25, minLineLength = edged.shape[:2][1]/10, maxLineGap = edged.shape[:2][1]/50) 
    # lines = cv2.HoughLinesP(orig_bin.copy(), rho = 1, theta = 3.14/180, threshold = 5000, minLineLength = orig_bin.shape[:2][1]/10, maxLineGap = orig_bin.shape[:2][1]/70) 
    # lines = cv2.HoughLinesP(closed_invert, rho = 1, theta = 3.14/180, threshold = 100, minLineLength = 0, maxLineGap = 0)
    # lines = cv2.HoughLinesP(edged, rho = 1, theta = 3.14/180, threshold = 10, minLineLength = 20, maxLineGap = 10) # this seems to work with the downsized image
    lines = []
    if hough_lines is None:
        print "NO LINES FOUND"
    else:    
        print "number of lines found " + str(hough_lines.shape)
        for hough_line in hough_lines:
            print hough_line
            for x1,y1,x2,y2 in hough_line:
                cv2.line(drawing_image,(x1,y1),(x2,y2),(255,0,0),1)
                line = ((x1, y1), (x2, y2))
                lines.append(line)
    return lines, drawing_image

def calculate_radians(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    denominator = (x2 - x1)
    # if denominator < 
    slope = ((y2 - y1) * 1.0) / ((x2 - x1) * 1.0)
    radians = math.atan(slope) # might want to return the radians rather than the degrees because the binning is more general
    # angle = math.degrees(radians)
    # print "calculating slope for " + str(pt1) + " and " + str(pt2) + ". Got " + str(slope) + " and angle " + str()
    return radians

def prob_hough_rotation(image, drawing_image):
    lines, drawn_image = prob_hough(image, drawing_image)
    radians = dict()
    for line in lines:
        pt1, pt2 = line
        x1, y1 = pt1
        x2, y2 = pt2
        radian = round(calculate_radians((x1, y1), (x2, y2)), 1)
        if radian in radians.keys(): radians[radian] = radians[radian] + 1
        else: radians[radian] = 1 
    freq_radian = max(radians, key=radians.get)
    # rotation_angle = (90 - math.degrees(freq_radian)) * -1
    rotation_angle = math.degrees(freq_radian)
    print radians
    print "Most frequent radians is: " + str(freq_radian) +  "Rotation angle is: " + str(rotation_angle)
    rotated_image = functions.rotate(drawn_image.copy(), rotation_angle)
    functions.plot_images([image, drawn_image, rotated_image], ["Input image", "HoughLinesP", "Rotated"]) 
    return rotated_image      

def prob_hough_display_rotated(images, drawing_images):
    if not len(drawing_images) == len(images):
        print "IN prob_hough_display, number of images and images to draw on are not the same"
        return 
    else:
        results = []
        for i, image in enumerate(images):
            rotated_image = prob_hough_rotation(image, drawing_images[i])
            results.append[rotated_image]
    functions.plot_images[results]


def std_hough_display(images):
    results = []
    for image in images:
        _, drawn_image, rotated_image = standard_hough(image, image)


def original(image_path_arg):
    orig = cv2.imread(sys.argv[1])

    # these constants are carefully picked
    MORPH = 9
    CANNY = 84
    HOUGH = 25

    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(img, (3,3), 0, img)

    # this is to recognize white on white
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
    dilated = cv2.dilate(img, kernel)

    edges = cv2.Canny(dilated, 0, CANNY, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1,  3.14/180, HOUGH)
    for line in lines[0]:
         cv2.line(edges, (line[0], line[1]), (line[2], line[3]),
                         (255,0,0), 2, 8)

    # finding contours
    # contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, # was this
    #                                cv2.CHAIN_APPROX_TC89_KCOS)

    ims, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, # changed to this
                                   cv2.CHAIN_APPROX_TC89_KCOS)                                   
    contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours)
    contours = filter(lambda cont: cv2.contourArea(cont) > 10000, contours)

    # simplify contours down to polygons
    rects = []
    for cont in contours:
        rect = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
        rects.append(rect)

    # that's basically it
    cv2.drawContours(orig, rects,-1,(0,255,0),1)

    # show only contours
    new = get_new(img)
    cv2.drawContours(new, rects,-1,(0,255,0),1)
    cv2.GaussianBlur(new, (9,9), 0, new)
    new = cv2.Canny(new, 0, CANNY, apertureSize=3)


def contouring(image):
        # <---- FIND CONTOURS ---> #
    # ims, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, # this works pretty well...mostly just grabs the document
    #                                cv2.CHAIN_APPROX_SIMPLE)                                   
    # ims, contours, hierarchy = cv2.findContours(closed_invert.copy(), cv2.RETR_TREE, # this doesn't work...grabs whole image
    #                                cv2.CHAIN_APPROX_SIMPLE)  
    # ims, contours, hierarchy = cv2.findContours(lined_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # latest effort after seeing that Hough Lines appears to work pretty well                                        
    ims, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = filter(lambda cont: cv2.arcLength(cont, False) > 100, contours) # was 100
    contours = filter(lambda cont: cv2.contourArea(cont) > 1000, contours) # was 10000

    # simplify contours down to polygons
    rects = []
    for cont in contours:
        # rect = cv2.approxPolyDP(cont, 40, True).copy().reshape(-1, 2)
        # rects.append(rect)
        peri = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
        rects.append(approx)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] # added the [:5] per the kickass scanner example
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    rect_detected = cv2.drawContours(image = orig_resized.copy(), contours = [box], contourIdx = 0, color = (0,255,0), thickness = 1) # was an idiot and had it drawing on the full size image (coordinates off)
    poly_detected = cv2.drawContours(image = orig_resized.copy(), contours = rects, contourIdx = -1, color = (0,255,0), thickness = 1)

    # functions.plot_images([orig, poly_detected, rect_detected])


if __name__ == '__main__':
    files = utility.image_paths() 
    originals = []
    images = []
    for filename in files:
        image = cv2.imread(filename)
        images.append(image)
        originals.append(image.copy())

    

    prob_hough_display_rotated(images, originals)

    orig = cv2.imread(sys.argv[1])
    # orig_bin = cv2.imread(sys.argv[1], cv2.CV_8UC1)
    orig_resized = functions.standard_resize(orig.copy())

    edged_resized = functions.colorOps(orig_resized) #grayscale -> gaussian blur -> canny
    # bordered_edged_resized = cv2.copyMakeBorder(edged_resized, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255, 255, 255])		 # added this 11/2/17. Trying to work with document occlusion. 
    # edged = functions.colorOps(orig) #grayscale -> gaussian blur -> canny. This working fairly well for isolating text lines in full size images, but not for sample1.
    
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, threshold1 = 0, threshold2 = 80, apertureSize=3, L2gradient=True)
    kernel = np.ones((5,5),np.uint8) # original was 15x15. 5x5 is working well right now. 
    dilated = cv2.dilate(edged, kernel, iterations=1) # THIS LOOKS GOOD ON ALL but s4!! It erodes too much though. Losing some of document.

    closedEdges = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel = np.ones((5, 11))) # this potentially helps close Canny edges that are close but not quite touching
    # functions.plot_images([dilated, closedEdges])

    # edged = functions.colorOps(orig) #grayscale -> gaussian blur -> canny

    # closed_invert = functions.closed_inversion(edged) # perform close -> invert (so document is white)

    # text = text_regions.text_regions(orig.copy()) # this works fairly well but don't have MSER in opencv.js. Canny on full size seems to work just as well.
    # edged = cv2.Canny(text, threshold1 = 75, threshold2 = 200) # was 0, 50...not sure what these numbers mean

    # lines, lined_image = prob_hough(image = edged, drawing_image = orig) # this is for text lines (use full size image)
    # lines, lined_image = prob_hough(image = dilated, drawing_image = orig) # this is for text lines (use full size image)
    # lines2, lined_image2 = prob_hough(image = closedEdges, drawing_image = orig) # this is for text lines (use full size image). This was the last attempt. 
    # std_lines, std_lined_image = standard_hough(image = edged_resized, drawing_image = orig_resized) 
    prob_hough_rotation(image = closedEdges, drawing_image = orig)
    
    # lines, lined_image = standard_hough(image = edged, drawing_image = orig) # this was last attempt
    # lines, lined_image = standard_hough(image = dilated, drawing_image = orig) # this was last attempt
    # lines, lined_image = standard_hough(image = closedEdges, drawing_image = orig)
        
    lined_bw = cv2.cvtColor(lined_image, cv2.COLOR_BGR2GRAY) # the problem with this is that the end result is no different than the binarized image from which the lines were found
    # basically, I don't see a benefit to using HoughLines, other than maybe to filter out noise. 
    # Revision: the benefit is that you can do computation on the lines returned, rather than relying on a blackbox method to find contours/rectangles in an image.

    # functions.plot_images([edged, dilated, lined_image], ["edged", "dilated_edged", "lined"])
    # rotated = functions.rotate(orig, 31.987)
