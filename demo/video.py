import cv2
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
import functions
import demo
import hough
# import experiments
import parser
import utility

import detection
import boxing
import hough_clean

import importlib
importlib.reload(functions)

# NOTES: text_region_method in demo.py is currently best method (actually only method that works). 
# Currently running around 10 fps. 
# Not working well on patent005.avi

# cap = cv2.VideoCapture(0) # this uses the webcam. pretty cool. 
# cap = cv2.VideoCapture('vids/background01/tax004.avi')
# print type(cap)

# videofile = 'data/vids/background00/datasheet001.avi' # DOES WELL HERE, BUT MISSES 33-40 DETECTIONS. NEED TO ADD 'SUCCESS' CHECK FROM CORNERING. 
# videofile = 'data/vids/background00/letter001.avi' # DOES WELL HERE. ONLY MISSES 1. 
# videofile = 'data/vids/background00/magazine001.avi' # performs terribly. Has trouble with clean edge detection. 
# videofile = 'data/vids/background01/patent002.avi' # video broken
# videofile = 'data/vids/background01/patent004.avi' # DOES WELL HERE. BUT MISSES 13 DETECTIONS. 
# videofile = 'data/vids/background01/patent005.avi' # PERFORMS FLAWLESSLY
# videofile = 'data/vids/background01/tax003.avi' # Performs pretty well given the amount of noise. Adjusted some HOUGH parameters, including tightened angle range and allowing consideration of more lines. 
# videofile = '/Users/echristiansen/Data/SmartDoc/challenge_1_seg/testDataset/background04/clips/datasheet003.avi'
videofile = 'data/vids/tax002.avi'
# videofile = 'data/vids/datasheet001.avi' # this one is HARD
# cap = skvideo.io.VideoCapture(video)
cap = skvideo.io.vread(videofile)
num_frames = cap.shape[0]
print(cap.shape)
video = skvideo.io.FFmpegReader(videofile)

# path, ext =  video.split('.')
# truth_file = path + '.gt.xml'
# truth = parser.get_truth(truth_file)
failed_detections = 0
frame_count = 0

# frame_width, frame_height = (1920, 1080)
frame_width, frame_height = (56, 100)

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width, frame_height))

for i, frame in enumerate(video.nextFrame()):
        frame_count = frame_count + 1

        # Capture frame-by-frame
        # ret, frame = cap.read()

        # title = "Frame " + str(count)
        # points = truth.get(count)
        # box = parser.create_box(points)
        # print points
        # print box

        # downsized, ratio = functions.standard_resize(frame, new_width = 100.0)

        # small_box = box[:,:] / ratio # this was for seeing if could draw ground truth on smaller image
        # small_box = np.round(small_box)
        # small_box = small_box.astype(int)
        # print small_box
        # print small_box.dtype

        
        # edged, titles = experiments.canny(downsized)
        # originals = utility.image_array(image = downsized, array_length = len(edged))
        # # functions.plot_images(edged, titles)
        # lined = functions.draw_several(images = edged[2:], drawing_images = originals[2:], function = hough.standard_hough)
        # functions.plot_images(edged + lined)

        # original = frame.copy()
        # hough.hough_cornering(frame)

        '''Was running this'''
        demo.detection(frame)

        # lined = hough.hough_video(frame)
        # lined = hough.hough_video(frame)
        # print(lined.shape)

        # resized = cv2.resize(lined, (frame_width, frame_height), interpolation = cv2.INTER_CUBIC)

        # out.write(resized)
        


        # '''INTIAL CODE'''
        # '''Trying Hough now'''
        # print('Now on frame ' + str(i))

        '''WAS HERE 4/25'''
        # orig = frame.copy()
        # cropped, box, worked = detection.detection(frame)
        # # corners = hough.hough_cornering(cropped, orig = orig, crop_box = box)
        # # corners, success = hough.hough_cornering_draw(image=cropped, orig=orig, crop_box=box) # was here

        # # crop_box = np.zeros(shape=(4,2))
        # corners, success = hough.hough_cornering_draw(image=cropped, orig=orig, crop_box=box) # Trying no noise reduction step
        # # corners = hough_clean.hough_detection(image = cropped)
        # print(corners)
        ''''''

        # if not worked:
                # failed_detections = failed_detections + 1    

        # if i == 37:
        #         cv2.imwrite('frame38.jpg', frame)

        # '''INTIAL CODE'''
        
        # experiments.downsized_canny_CI(frame)
        
        # experiments.downsized_canny_detection(frame)

        # edged, titles = experiments.canny_thresh(downsized)
        # functions.plot_images(edged, titles)


        # correct = cv2.drawContours(downsized,[small_box],0,(255,0,0),1)
        # functions.plot_images([correct], [title])

        # demo.occlusion_demo(frame)  # NOT WORKING AT ALL
        # demo.hull_attempt(frame)  # NOT WORKING AT ALL
        # demo.pipeline(frame)  # NOT WORKING AT ALL
        # demo.text_region_method(frame) # WORKING FAIRLY WELL
        # experiments.canny_comparison(frame)

        # experiments.downsized_canny_detection(frame) # WAS HERE 12/3 LATE


        # experiments.downsized_

        # demo.standard_hough(frame)
        # demo.downsized_text_blobbing(frame) # NOT WORKING AT ALL
        # functions.plot_image(frame, title)

print("Number of failed detections: " + str(failed_detections) + " out of " + str(frame_count) + " total frames")