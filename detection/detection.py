import cv2
import sys
import numpy as np
import argparse
import imutils
import math
from matplotlib import pyplot as plt
from skimage.filters import threshold_adaptive
import functions
import utility
import hough
import time

import edging
import coloring
import reshape
import boxing

# <---- DEMO METHODS -----> #

def occlusion_demo(image):
    orig = image.copy()

    # <---- RESIZING -----> #
    image, ratio = functions.standard_resize(image, new_width = 100.0)
    # <---- RESIZING -----> #

    edged = functions.colorOps(image)
    processed = functions.closed_inversion(edged)
    points = functions.minRectMethod(processed)
    imutils.negative_coords(points, processed.shape[1], processed.shape[0])
    # points = functions.minRectMethod(edged)
    detection = cv2.drawContours(image.copy(), [points], -1, (0, 255, 0), 2)
    final = functions.finalize(orig.copy(), points, ratio)
    
    # cv2.imwrite('processed/final.jpg', final)
    functions.plot_images([orig, edged, processed, detection, final], ["Original", "Edge Detection", "Morpohological Operations", "Contour Finding", "Perspective Transform"])
    

def original_demo(image):
    orig = image.copy()
    
    # <---- RESIZING -----> #
    image, ratio = functions.standard_resize(image, new_width = 100.0)
    # <---- RESIZING -----> #

    processed = functions.colorOps(image)
    points = functions.contour_method(processed)
    detection = cv2.drawContours(image.copy(), [points], -1, (0, 255, 0), 2)
    final = functions.finalize(orig, points, ratio)
    
    images = [orig, detection, final]
    functions.plot_images(images)

# <---- END ORIGINAL DEMO METHODS -----> #


def detection_cropping(image):
    original = image.copy()
    downsized, ratio = reshape.standard_resize(image, new_width = 100.0)
    downsized = coloring.gamma_correction(image=downsized, correction=2)
    # edged = edging.orig_page_edging(image=downsized)
    merged = boxing.edged_to_boxes(image=downsized, edging_function = functions.orig_page_edging)
    # merged_boxes = boxing.generate_boxes(image=edged)

    box = np.zeros(shape=(4,2))

    if not type(merged) == np.ndarray:
        # edged = edging.page_edging(image=downsized, thresh1=0, thresh2=160)
        # merged = boxing.generate_boxes(image=edged)
        merged = boxing.edged_to_boxes(image=downsized, edging_function = functions.page_edging, thresh1=0, thresh2=160)
        if not type(merged) == np.ndarray:
            # edged = edging.page_edging(image=downsized, thresh1=0, thresh2=120)
            # merged = boxing.generate_boxes(image=edged)
            merged = boxing.edged_to_boxes(image = downsized, edging_function = functions.page_edging, thresh1=0, thresh2=120)
            if not type(merged) == np.ndarray:
                return original, box, False
    
    box = merged.reshape(4,2) * ratio
    box = np.int0(box)
    cropped = boxing.box_crop(original, box)
    return cropped, box, True

def detection(image):
    original = image.copy()
    downsized, ratio = reshape.standard_resize(image, new_width = 100.0)
    
    # downsized = functions.shadow_removal(image=downsized)
    downsized = coloring.gamma_correction(image=downsized, correction=2)
    # downs = functions.masking_attempt(image=original)
    
    merged, boxes, edged, closed_invert = boxing.boxes_from_edged(downsized, edging_function = functions.orig_page_edging)
    # merged, boxes, edged, closed_invert = boxes_from_edged(image=downsized, edging_function = functions.page_edging, thresh1=75, thresh2=220)

    # utility.plot_images([edged, closed_invert], titles = ["first pass", "first pass"])
    detected = image.copy()
    box = np.zeros(shape=(4,2))
    
    missed_detections = 0
    if not type(merged) == np.ndarray:
        # print("GOT NO BOXES, TRYING SMALL PAGE EDGING")
        # merged, boxes, edged, closed_invert = boxes_from_edged(downsized, edging_function = functions.small_page_edging)
        merged, boxes, edged, closed_invert = boxing.boxes_from_edged(downsized, edging_function = functions.page_edging, thresh1=0, thresh2=160)
        # utility.plot_images([edged, closed_invert], titles = ["second pass", "second pass"])
    
        if not type(merged) == np.ndarray:
            # print("GOT NO BOXES, TRYING SMALLER!! PAGE EDGING")
            # merged, boxes, edged, closed_invert = boxes_from_edged(downsized, edging_function = functions.smaller_page_edging)
            merged, boxes, edged, closed_invert = boxing.boxes_from_edged(downsized, edging_function = functions.page_edging, thresh1=0, thresh2=120)
            # utility.plot_images([edged, closed_invert], titles = ["third pass", "third pass"])
    
    if type(merged) == np.ndarray:
            # print("DRAWING THE BOX")
            # detected = cv2.drawContours(downsized.copy(), contours = boxes, contourIdx = -1, color = (0,255,0), thickness = 1) # if passing in a list and want to draw more than 1
            detected = cv2.drawContours(downsized.copy(), contours = [merged], contourIdx = -1, color = (0,255,0), thickness = 1) # if just drawing one (using merge boxes)
            all_boxes = cv2.drawContours(downsized.copy(), contours = boxes, contourIdx = -1, color = (255,0,0), thickness = 1) # if just drawing one (using merge boxes)
    else: 
        print("FOUND NO BOXES AT ALL!!!!")
        utility.plot_images([original, edged, closed_invert, detected], titles = ["original", "Edged", "closed_invert", "detected"])
        return original, box, False
    # warped = functions.perspective_transform(original, boxes, ratio)
    box = merged.reshape(4,2) * ratio
    box = np.int0(box)

    cropped = boxing.box_crop(original.copy(), box)
    downsize_cropped = reshape.standard_resize(image = cropped, new_width = 100.0, return_ratio = False)

    '''Was the idea here to potentially take a second pass at this now-cropped image? Just with Hough, or with same approach? Or use prob_hough_rotation?'''
    blob_cropped = edging.text_blobbing(downsize_cropped.copy())
    edge_cropped = edging.text_edging(downsize_cropped.copy()) 
    edge_cropped2 = edging.downsized_text_edging(downsize_cropped.copy())
    
    # lines = hough.standard_hough_lines(edged.copy())
    # corners = hough.hough_corners(lines)
    # cornered = hough.draw_corners(downsized.copy(), corners)

    '''Commenting this out to test negative check'''
    # utility.plot_images([original, edged, closed_invert, all_boxes, detected, cropped, edge_cropped, edge_cropped2], titles = ["original", "Edged", "closed_invert", "all boxes", "detected", "cropped", "edge_cropped", "edge_cropped2"])
    # return detected
    return cropped, box, True

def pipeline(image):
    orig = image.copy()
    downsized, ratio = functions.standard_resize(image, new_width = 100.0)
    edged = functions.small_page_edging(downsized)
    processed = functions.closed_inversion(edged)
    boxes = functions.box_generation(processed)
    filtered_boxes = functions.box_filtration(boxes)

    detected = downsized.copy()
    rects=[]
    if not filtered_boxes:
        # print "FOUND NO BOXES; TRYING DIFFERENT CANNY"
        # edged = functions.text_edging(orig.copy())
        # edged = functions.downsized_text_edging(downsized.copy())
        edged = functions.smaller_page_edging(downsized)
        # rotated = hough.prob_hough_rotation(edged, orig.copy())
        # detected = rotated
        processed = functions.closed_inversion(edged)
        boxes = functions.box_generation(processed)
        filtered_boxes = functions.box_filtration(boxes)
        final_box = functions.merge_boxes(filtered_boxes)
        if final_box:
            # final_box = final_box * ratio
            final_box = box[:,:] * ratio
            final_box = np.round(small_box)
            final_box = small_box.astype(int)

            warped = functions.perspective_transform(orig.copy(), final_box, ratio = ratio)
            lined = hough.standard_hough(warped)
        else:
            print("in demo pipeline")

    else:            
        for box in boxes:
            # print box
            detected = cv2.drawContours(detected,[box],0,(0,255,0),1)
            rects.append(cv2.minAreaRect(box))
            # print "rect in alternate_rect_attempt: " + str(cv2.minAreaRect(box))
        if len(boxes) > 1:
            # print "got more than 1 box, attempting merge"
            merged = functions.merge_boxes(boxes)
            detected = cv2.drawContours(detected,[merged],0,(255,0,0),2)
    functions.plot_images([edged, processed, detected], ["Edge Detection", "Morphological Operations", "Contour Finding"])
    return detected


def hough_blobbing(image): # doesn't really work. 
    orig = image.copy()
    # <---- RESIZING -----> #
    downsized, ratio = functions.standard_resize(image, new_width = 200.0)
    # <---- RESIZING -----> #
    edged = functions.downsized_text_edging(downsized.copy())
    # blank = np.ones(edged.shape[:3], np.uint8) * 255
    blank = np.zeros(edged.shape[:3], np.uint8)
    lines, drawn = hough.prob_hough(edged, blank)
    
    detected = downsized.copy()
    # boxes = functions.alternateRectMethod(drawn)
    boxes = functions.all_boxes(drawn)
    for box in boxes:
        detected = cv2.drawContours(detected,[box],0,(0,255,0),3)
    

    kernel = np.ones((3,3),np.uint8) # original 9x9
    dilated = cv2.dilate(drawn, kernel, iterations=5) # original was 5 iterations
    dilated = functions.closed_inversion(dilated)

    detected2 = downsized.copy()
    # boxes = functions.alternateRectMethod(dilated)
    boxes = functions.all_boxes(dilated)
    for box in boxes:
        detected2 = cv2.drawContours(detected2,[box],0,(0,255,0),3)

    functions.plot_images([downsized, edged, drawn, dilated, detected, detected2])

def text_blobbing(image):
    orig = image.copy()
    small_orig = functions.standard_resize(image, new_width = 100.0, return_ratio = False)

    start = time.clock()
    # downsized, ratio = functions.standard_resize(image, new_width = 100.0)

    edged = functions.text_edging(orig.copy())
    # downsized, ratio = functions.standard_resize(edged, new_width = 200.0)
    # edged = functions.downsized_text_edging(downsized)
    # dilated = functions.closed_inversion(downsized)
    # kernel = np.ones((3,3),np.uint8) # was 9x9
    # dilated = cv2.dilate(edged, kernel, iterations=3) 
    kernel = np.ones((13,13),np.uint8) # original 9x9
    dilated = cv2.dilate(edged, kernel, iterations=7)
    # erosion = cv2.erode(dilated, kernel, iterations=2) # DOESN'T WORK ON VIDEO

    # functions.plot_images([dilated])
    # boxes = functions.alternateRectMethod(edged)
    boxes = functions.box_generation(dilated)
    # boxes = functions.pureRectMethod(dilated) # WAS USING THIS ON VIDEOS WORKING FAIRLY WELL
    # boxes = functions.pureRectMethod(erosion)
    
    # detected = orig.copy()
    box = []
    detected = small_orig.copy()

    if not boxes:
        # print "NO BOXES FOUND"
        final = detected
    else:   
        detected = cv2.drawContours(orig,boxes,0,(0,255,0),3)
        box = boxes[0]
        final = functions.perspective_transform(orig.copy(), box)
        # for box in boxes:
        #     detected = cv2.drawContours(detected,[box],0,(0,255,0),1)
        #     box = boxes[0]
        #     final = functions.perspective_transform(orig.copy(), box)
    end = time.clock()
    # print "Frame took " + (str (end-start)) + " time to process"

    functions.plot_images([orig, dilated, detected, final], ["orig", "dilated", "text_region_method", "final"])
    # cv2.imwrite(filename = "iou_test.png", img = detected)
    # functions.plot_images([erosion, detected], ["dilated", "text, _region_method"])


def hull_attempt(image): # these methods seem ripe for command pattern (or strategy?) and/or just using Python's ability to pass methods in parameters
    orig = image.copy()
    
    # <---- RESIZING -----> #
    image, ratio = functions.standard_resize(image, new_width = 100.0)
    # <---- RESIZING -----> #

    edged = functions.colorOps(image)

    # <---- TRIED THIS BUT DIDN'T WORK WELL -----> #
    # points = hullMethod(processed)
    # detection = cv2.drawContours(image = image.copy(), contours = points, contourIdx = -1, color = (0, 255, 0), thickness = 1)

    # for point in points:
    #     final = functions.functions.finalize(orig, point, ratio)
    
    # images = [orig, detection, final]
    # functions.plot_images(images)
    # <---- SWITCHED TO THE CODE BELOW AND IT WORKS PRETTY WELL. NEED TO FILTER ON SIZE OR SOMETHING, TOO MANY BOXES-----> #

    processed = functions.closed_inversion(edged)
    functions.plot_images([edged, processed])
    points = functions.hullRectMethod(processed)

def alternate_rect_attempt(image):
    orig = image.copy()
    
    # <---- RESIZING -----> #
    image, ratio = functions.standard_resize(image, new_width = 100.0)
    # <---- RESIZING -----> #

    # edged = functions.colorOps(image)
    edged = functions.page_edging(image)
    processed = functions.closed_inversion(edged)
    boxes = functions.alternateRectMethod(processed)
    functions.boxes_comparison(image = processed, boxes = boxes)

    detected = image.copy()
    rects=[]
    for box in boxes:
        # print box
        detected = cv2.drawContours(detected,[box],0,(0,255,0),1)
        rects.append(cv2.minAreaRect(box))
        # print "rect in alternate_rect_attempt: " + str(cv2.minAreaRect(box))
    # utility.IOU(rects[0], rects[1])
    if len(boxes) > 1:
        # print "got more than 1 box, attempting merge"
        merged = functions.merge_boxes(boxes)
        detected = cv2.drawContours(detected,[merged],0,(255,0,0),2)
    functions.plot_images([edged, processed, detected], ["Edge Detection", "Morphological Operations", "Contour Finding"])


def text_region_method2(image):
    orig = image.copy()

    edged = functions.text_edging(orig.copy())
    kernel = np.ones((9,9),np.uint8) # original 9x9
    dilated = cv2.dilate(edged, kernel, iterations=7) # original was 5 iterations. This works a little better?

    points = functions.minRectMethod(dilated)
    detection = cv2.drawContours(image.copy(), [points], -1, (0, 255, 0), 2)
    final = functions.finalize(orig.copy(), points)
    functions.plot_images([orig, edged, dilated, detection, final], ["Original", "Edged", "Dilated", "Detection", "Final"])


def downsized_text_blobbing(image):
    orig = image.copy()
    
    # <---- RESIZING -----> #
    image, ratio = functions.standard_resize(image, new_width = 150.0)
    # <---- RESIZING -----> #

    # edged = functions.text_edging(orig.copy())
    # edged = functions.text_edging(image.copy())
    edged = functions.downsized_text_edging(image.copy())
    kernel = np.ones((3,3),np.uint8) # original 9x9
    dilated = cv2.dilate(edged, kernel, iterations=3) # original was 5 iterations

    points = functions.box_generation(dilated)
    # print "# printing points in text_region_method2" + str(points)
    points = points[0] * ratio
    # print "# printing points in text_region_method2" + str(points)
    # detection = cv2.drawContours(image.copy(), [points], -1, (0, 255, 0), 2)
    # final = functions.finalize(orig.copy(), points)
    final = functions.perspective_transform(orig.copy(), points)
    functions.plot_images([orig, edged, dilated, final], ["Original", "Edged", "Dilated", "Final"])


if __name__ == '__main__':
    # image = cv2.imread(sys.argv[1])
    # image_area = image.shape[0] * image.shape[1]
    # print "initial image area is: "

    files = ["pics/forms/sample4_3.jpg", "pics/forms/sample4_4.jpg", "pics/invoice2/sample1.JPG", "pics/invoice2/sample1_2.JPG"]
    files = ["pics/forms/sample5.jpg", "pics/forms/sample2.jpg", "pics/forms/sample3.jpg", "pics/forms/sample4_4.jpg", "pics/forms/sample9.jpg", "pics/forms/sample11.jpg", "pics/forms/sample8.jpg", "pics/forms/sample12.jpg"]
    # files = ["pics/forms/sample5.jpg", "pics/forms/sample2.jpg", "pics/forms/sample4_4.jpg", "pics/forms/sample11.jpg", "pics/forms/sample12.jpg", "pics/forms/sample10.jpg"]
    files = ["data/pics/demo/IMAG0603.jpg", "data/pics/demo/IMAG0604.jpg", "data/pics/demo/IMAG0605.jpg", "data/pics/demo/IMAG0606.jpg", "data/pics/demo/IMAG0607.jpg", "data/pics/demo/IMAG0608.jpg", "data/pics/demo/IMAG0611.jpg", "data/pics/demo/IMAG0612.jpg"]
    # files = ["pics/forms/E-Ticketing.png", "pics/indoor_720.jpg", "pics/forms/sample4_2.jpg"]

    # files = utility.filenames_at_path("/media/thor/LEXAR/sampleDataset/input_sample", ".jpg") # hough threshold of 30 is best here
    # files = utility.filenames_at_path("/home/thor/code/sr_project/pics/forms", ".jpg")

    originals, images = utility.image_reading(files[:20])

    for image in images:
        detection(image.copy())
        # text_region_method(image.copy())
        # downsized_text_blobbing(image.copy())


    originals, images = utility.image_reading(files)
    # results = functions.process_several(images, function = pipeline)
    # results = functions.process_several(images, function = downsized_text_blobbing)
    results = functions.process_several(images, function = text_region_method)
    functions.plot_images(results, files)
    # text_region_method2(image)
    # text_region_method(image)
    # downsized_text_blobbing(image)
    
    # canny_comparison(image)
    # dilation_canny(image)
    # rotate(image, 20)
    # occlusion_demo(image)
    # original_demo(image)
    
    hull_attempt(image)
    
    # alternate_rect_attempt(image)
    
    # prob_hough(image)

    # hough_blobbing(image)
    # new_docs(image)
    # blur_comparison(image)
    # edging_comparisons(image)
