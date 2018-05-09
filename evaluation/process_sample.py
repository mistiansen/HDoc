#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple detection program for SmartDOC challenge 1: 
    ``Document object detection and segmentation in preview frames''

A quick way to start is to check and change the implementation for the 3 TODOs
below.

To invoke this program, use:
$ python process_sample.py /path/to/backgroundXX/documentNNN.avi -o /path/to/xml/output

To see invocation options, use:
$ python process_sample.py -h

You will need a working installation of:
- Python (>2.7)
- Numpy
- OpenCV (>2.4.8) with XVID support

"""

# ==============================================================================
# Imports
import logging
import argparse
import os
import os.path
import sys
import datetime
import cv2
import numpy as np
import skvideo.io
import functions

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# from detection import detection 
import detection
import hough
import hough_clean
import boxing
import imutils
import time


# ==============================================================================
logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
PROG_VERSION = "0.1"
PROG_NAME = "Simple detection program"

EXITCODE_OK = 0
EXITCODE_KBDBREAK = 10
EXITCODE_UNKERR = 254

# ==============================================================================
class FrameIterator(object):
    """
    Wrapper for OpenCV video file reader as a Python iterator.
    """
    def __init__(self, videofile):
        if not os.path.exists(videofile):
            err = "'%s' does not exist." % videofile
            logger.error(err)
            raise IOError(err)
        # self._videocap = cv2.VideoCapture(videofile)
        # self._frame_count = int(self._videocap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        # self._videocap = skvideo.io.vread(videofile)
        self._videocap = skvideo.io.FFmpegReader(videofile)
        self._frame_count = skvideo.io.vread(videofile).shape[0]
        print(self._frame_count)
        logger.debug("Input video informations:")
        logger.debug("\tframe_count = %d" % self._frame_count)
        self._cfid = 0
        self._prevRes = True

    def __iter__(self):
        return self

    def __next__(self):
        if self._prevRes and self._cfid < self._frame_count:
            # self._prevRes, frame = self._videocap.read()
            frame = self._videocap.nextFrame()
            print("Frame is " + str(frame))
            self._cfid += 1
            if self._prevRes:
                return (self._cfid, frame)
        # else
        raise StopIteration

    def next(self):
        return self.__next__()

    def release(self):
        self._videocap.release()
        self._videocap = None

# ==============================================================================
class OutputDriver(object):
    """
    Very simple driver for XML result output.
    """
    def __init__(self):
        self.source_sample_file = ""
        self.software_used = ""
        self.software_version = ""
        self.segmentation_results = []

    def exportToFile(self, xmlfile):
        with open(xmlfile, "w") as out:
            out.write("<?xml version='1.0' encoding='utf-8'?>\n")
            out.write("<seg_result version=\"0.2\" generated=\"%s\">\n" % datetime.datetime.now().isoformat())
            out.write("  <software_used name=\"%s\" version=\"%s\"/>\n" % (self.software_used, self.software_version))
            out.write("  <source_sample_file>%s</source_sample_file>\n" % self.source_sample_file)
            out.write("  <segmentation_results>\n")
            for (fidx, rejected, tl, bl, br, tr, process_time) in self.segmentation_results: # ADDED SPEED
                if rejected:
                    print("EXPORTING A REJECTION TO XML")
                    # out.write("    <frame index=\"%d\" rejected=\"true\"/>\n" % fidx) # WAS JUST THIS
                    out.write("    <frame index=\"%d\" rejected=\"true\">\n" % fidx) # added this 
                    out.write("       <speed secs=\"%f\"/>\n" % process_time) # added this 
                    out.write("    </frame>\n") # SHOULD THIS HAPPEN REGARDLESS OF REJECTED? # added this
                else:
                    out.write("    <frame index=\"%d\" rejected=\"false\">\n" % fidx)
                    out.write("       <point name=\"bl\" x=\"%f\" y=\"%f\"/>\n" % (bl[0], bl[1]))
                    out.write("       <point name=\"tl\" x=\"%f\" y=\"%f\"/>\n" % (tl[0], tl[1]))
                    out.write("       <point name=\"tr\" x=\"%f\" y=\"%f\"/>\n" % (tr[0], tr[1]))
                    out.write("       <point name=\"br\" x=\"%f\" y=\"%f\"/>\n" % (br[0], br[1]))
                    out.write("       <speed secs=\"%f\"/>\n" % process_time) # ADDED THIS
                    out.write("    </frame>\n") # SHOULD THIS HAPPEN REGARDLESS OF REJECTED?
            out.write("  </segmentation_results>\n")
            out.write("</seg_result>\n")


# ==============================================================================
class Tracker(object):
    """
    A Tracker can process frames and find objects inside them.
    """

    def processFrame(self, frame_id, image, speed = False):
        """
        Will be called once with each frame to process.
        """

        start = time.clock()

        cropped, crop_box, crop_worked = detection.detection_cropping(image=image)
        # corners, successful = hough_clean.hough_detection(image=cropped)
        
        orig = image.copy()
        corners, successful = hough.hough_cornering(cropped, orig = orig, crop_box = crop_box)

        rejected = False
        if not successful and not crop_worked:
            rejected = True
            print("REJECTED")
            tl, tr, bl, br = 0, 0, 0, 0
        elif not successful and crop_worked:
            box = imutils.order_points(pts = crop_box)
            tl, tr, br, bl = box
            # print("ONLY USING CROP REGION AS DETECTION")
        else:
            # xmin, xmax, ymin, ymax = boxing.max_points(crop_box)
            # corners[:, 0] += xmin
            # corners[:, 1] += ymin
            tl, tr, br, bl = corners
            # print("Final tl: " + str(tl))
            # print("Final tr: " + str(tr))
            # print("Final br: " + str(br))
            # print("Final bl: " + str(bl))

        end = time.clock()
        process_time = end - start

        # return (rejected, tl, bl, br, tr) # original
        if speed:
            return (tl, tr, br, bl), rejected, process_time
        else:
            return (tl, tr, br, bl), rejected


# ==============================================================================


def process_samples(argv):

    background = 'background01'
    input_path = '/Users/echristiansen/Data/SmartDoc/challenge_1_seg/testDataset/' + background + '/clips/'
    output_path = '/Users/echristiansen/Data/SmartDoc/challenge_1_seg/results/test/' + background + '/'
    video_out_path = '/Users/echristiansen/Data/SmartDoc/challenge_1_seg/results/vids/' + background + '/'
    for filename in os.listdir(path = input_path):
        if filename.endswith('.avi'):
            document = filename.split('.')[0]
            input_sample = input_path + filename
            output_file = output_path + document + '.xml'
            frames, tracker, output = init_processing(args=args, input_sample=input_sample)
            for (fidx, mat) in frames:
                frame = next(mat) # NEW for Python 3+?
                (frame_height, frame_width, _ch) = frame.shape 
                box, rejected, process_time = tracker.processFrame(fidx, frame, speed=True)
                tl, tr, br, bl = box
                if not rejected:
                    logger.info("frame %04d: A tl:(%-4.2f,%-4.2f) bl:(%-4.2f,%-4.2f) br:(%-4.2f,%-4.2f) tr:(%-4.2f,%-4.2f)" 
                                %(fidx, tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1]))
                else:
                    logger.info("frame %04d: R" % fidx)
                
                '''APPEND RESULT TO OUTPUT'''
                # output.segmentation_results.append((fidx, rejected, tl, bl, br, tr))
                output.segmentation_results.append((fidx, rejected, tl, bl, br, tr, process_time))

            '''WRITE ALL RESULTS TO XML FILE'''
            output.exportToFile(xmlfile = output_file)
            logger.debug("SegResult file generated: %s" % args.output_file)
        else:
            print("Encountered non-AVI file in path " + filename)
            continue
          # --------------------------------------------------------------------------
        logger.debug("--- Process complete. ---")


def init_processing(args, input_sample):
        # -----------------------------------------------------------------------------
    # Logger init
    format="%(name)-12s %(levelname)-7s: %(message)s" #%(module)-10s
    formatter = logging.Formatter(format)    
    ch = logging.StreamHandler()  
    ch.setFormatter(formatter)  
    logger.addHandler(ch)
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG
    logger.setLevel(level)
    logger.debug("Arguments:")
    for (k, v) in args.__dict__.items():
        logger.debug("    %-20s = %s" % (k, v))

    # --------------------------------------------------------------------------
    # Prepare process
    logger.debug("Starting up")
    # frames = FrameIterator(args.input_sample)
    frames = FrameIterator(input_sample)

    # --------------------------------------------------------------------------
    # Initialize tracker
    logger.debug("Creating tracker...")
    tracker = Tracker()

    # --------------------------------------------------------------------------
    # Output preparation
    output = OutputDriver()
    # output.source_sample_file = args.input_sample
    output.source_sample_file = input_sample
    output.software_used = "EHC tracker" # TODO change name
    output.software_version = "0.1" # TODO change version

    # Let's go
    # --------------------------------------------------------------------------
    logger.debug("--- Process started. ---")
    # --------------------------------------------------------------------------
    return frames, tracker, output


# ==============================================================================



def process_background(background, save_videos = False):

    # background = 'background01' # just need to change this to test different backgrounds
    input_path = '/Users/echristiansen/Data/SmartDoc/challenge_1_seg/testDataset/' + background + '/clips/'
    output_path = '/Users/echristiansen/Data/SmartDoc/challenge_1_seg/results/test/' + background + '/'
    video_out_path = '/Users/echristiansen/Data/SmartDoc/challenge_1_seg/results/vids/' + background + '/'

    for filename in os.listdir(path = input_path):
        if filename.endswith('.avi'):
            document = filename.split('.')[0]
            input_sample = input_path + filename
            output_file = output_path + document + '.xml'
            # original_process(input_sample=input_sample, output_file=output_file, gui=True)
            original_process(input_sample=input_sample, output_file=output_file, video_save_path=video_out_path, video=save_videos, gui=False)
        else:
            print("Encountered non-AVI file in path " + filename)
            continue
          # --------------------------------------------------------------------------
        logger.debug("--- Process complete. ---")



def original_process(input_sample, output_file, video_save_path, video = True, gui=True, debug=False):
    # -----------------------------------------------------------------------------
    # Logger init
    format="%(name)-12s %(levelname)-7s: %(message)s" #%(module)-10s
    formatter = logging.Formatter(format)    
    ch = logging.StreamHandler()  
    ch.setFormatter(formatter)  
    logger.addHandler(ch)
    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logger.setLevel(level)
    logger.debug("Arguments:")
    # for (k, v) in args.__dict__.items():
    #     logger.debug("    %-20s = %s" % (k, v))

    # --------------------------------------------------------------------------
    # Prepare process
    logger.debug("Starting up")
    frames = FrameIterator(input_sample)

    # --------------------------------------------------------------------------
    # Initialize tracker
    logger.debug("Creating tracker...")
    tracker = Tracker()

    # --------------------------------------------------------------------------
    # Output preparation
    output = OutputDriver()
    output.source_sample_file = input_sample
    output.software_used = "My simple tracker" # TODO change name
    output.software_version = "0.1" # TODO change version

    # Let's go
    # --------------------------------------------------------------------------
    logger.debug("--- Process started. ---")
    # --------------------------------------------------------------------------
    win_name = 'Tracker output'
    if gui:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    if video:
        '''Added this to write output videos'''    
        input_filename = input_sample.split('/')[-1] # input filename is the last element of the path string
        video_prefix = input_filename.split('.')[0] # result: e.g., datasheet001
        video_name = video_prefix + '.avi'
        video_full_path = video_save_path + video_name
        video_out = cv2.VideoWriter(video_full_path, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1920, 1080))

    for (fidx, mat) in frames:
        frame = next(mat) # NEW for Python 3+?
        (frame_height, frame_width, _ch) = frame.shape 
        box, rejected, process_time = tracker.processFrame(fidx, frame, speed=True)
        tl, tr, br, bl = box

        # if not rejected:
        #     logger.info("frame %04d: A tl:(%-4.2f,%-4.2f) bl:(%-4.2f,%-4.2f) br:(%-4.2f,%-4.2f) tr:(%-4.2f,%-4.2f)" 
        #                 %(fidx, tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1]))
        # else:
        #     logger.info("frame %04d: R" % fidx)
        if rejected:
            logger.info("frame %04d: R" % fidx)

        # Debug viz
        # if gui:
        if not rejected and (video or gui):
            dgbq = np.int32([[tl[0], tl[1]],
                                [bl[0], bl[1]],
                                [br[0], br[1]],
                                [tr[0], tr[1]]])
            cv2.polylines(img=frame, pts=[dgbq], isClosed=True, color=(0, 255, 0), thickness=2)

            for txt, pt in zip(("tl", "bl", "br", "tr"), (tl, bl, br, tr)):
                cv2.putText(frame, txt.upper(), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 2, (64, 255, 64), 2)
        else:
            cv2.circle(frame, (int(frame_width/2), int(frame_height/2)), 20, (0, 0, 255), 10) 

        if gui:
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) # required otherwise display thread show nothing
            if key & 0xFF == ord('q'):
                logger.warning("Exit requested, breaking up…")
                return EXITCODE_KBDBREAK

        '''Added this to write output videos'''
        if video:    
            video_out.write(frame)
        
        '''OUTPUT TO XML FILE'''
        output.segmentation_results.append((fidx, rejected, tl, bl, br, tr, process_time))

    # Output file
    if output_file is not None:
        output.exportToFile(output_file)
        logger.debug("SegResult file generated: %s" % output_file)

    # --------------------------------------------------------------------------
    logger.debug("--- Process complete. ---")
    # --------------------------------------------------------------------------
    if gui:
        logger.info("Press any key to exit.")
        cv2.waitKey()
        cv2.destroyWindow(win_name)


# ==============================================================================


def main(argv):
    # Option parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Detects and segments document objects in frames.')
        # ,version=PROG_VERSION)

    parser.add_argument('-d', '--debug', 
        action="store_true", 
        help="Activate debug output.")

    parser.add_argument('-g', '--gui', 
        action="store_true", 
        help="Activate GUI to visualize tracker output.")

    parser.add_argument('-o', '--output-file', 
        help="Optionnal path to output file (XML description of segmentation).")

    parser.add_argument('input_sample', 
        help='Input sample.')

    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Logger init
    format="%(name)-12s %(levelname)-7s: %(message)s" #%(module)-10s
    formatter = logging.Formatter(format)    
    ch = logging.StreamHandler()  
    ch.setFormatter(formatter)  
    logger.addHandler(ch)
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG
    logger.setLevel(level)
    logger.debug("Arguments:")
    for (k, v) in args.__dict__.items():
        logger.debug("    %-20s = %s" % (k, v))

    # --------------------------------------------------------------------------
    # Prepare process
    logger.debug("Starting up")
    frames = FrameIterator(args.input_sample)

    # --------------------------------------------------------------------------
    # Initialize tracker
    logger.debug("Creating tracker...")
    tracker = Tracker()

    # --------------------------------------------------------------------------
    # Output preparation
    output = OutputDriver()
    output.source_sample_file = args.input_sample
    output.software_used = "My simple tracker" # TODO change name
    output.software_version = "0.1" # TODO change version

    # Let's go
    # --------------------------------------------------------------------------
    logger.debug("--- Process started. ---")
    # --------------------------------------------------------------------------
    win_name = 'Tracker output'
    if args.gui:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    for (fidx, mat) in frames:
        # frame = mat.next() # OLD for Python 2.+?
        frame = next(mat) # NEW for Python 3+?
        # functions.plot_images([frame], [fidx]) # added
        (frame_height, frame_width, _ch) = frame.shape 
        box, rejected, process_time = tracker.processFrame(fidx, frame, speed=True)
        tl, tr, br, bl = box

        if not rejected:
            logger.info("frame %04d: A tl:(%-4.2f,%-4.2f) bl:(%-4.2f,%-4.2f) br:(%-4.2f,%-4.2f) tr:(%-4.2f,%-4.2f)" 
                        %(fidx, tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1]))
        else:
            logger.info("frame %04d: R" % fidx)

        # Debug viz
        if args.gui:
            if not rejected:
                    dgbq = np.int32([[tl[0], tl[1]],
                                     [bl[0], bl[1]],
                                     [br[0], br[1]],
                                     [tr[0], tr[1]]])
                    # cv2.polylines(img=mat, pts=[dgbq], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.polylines(img=frame, pts=[dgbq], isClosed=True, color=(0, 255, 0), thickness=2)

                    for txt, pt in zip(("tl", "bl", "br", "tr"), (tl, bl, br, tr)):
                        # cv2.putText(mat, txt.upper(), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 2, (64, 255, 64), 2)
                        cv2.putText(frame, txt.upper(), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 2, (64, 255, 64), 2)
            else:
                # cv2.circle(mat, (frame_width/2, frame_height/2), 20, (0, 0, 255), 10) # was this. not sure why. 
                cv2.circle(frame, (frame_width/2, frame_height/2), 20, (0, 0, 255), 10) 
                
            # cv2.imshow(win_name, mat)
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) # required otherwise display thread show nothing
            if key & 0xFF == ord('q'):
                logger.warning("Exit requested, breaking up…")
                return EXITCODE_KBDBREAK

        '''OUTPUT TO XML FILE'''
        output.segmentation_results.append((fidx, rejected, tl, bl, br, tr))

    # Output file
    if args.output_file is not None:
        output.exportToFile(args.output_file)
        logger.debug("SegResult file generated: %s" % args.output_file)

    # --------------------------------------------------------------------------
    logger.debug("--- Process complete. ---")
    # --------------------------------------------------------------------------
    if args.gui:
        logger.info("Press any key to exit.")
        cv2.waitKey()
        cv2.destroyWindow(win_name)


# ==============================================================================
# ==============================================================================
if __name__ == "__main__":

    background = 'background05'
    process_background(background=background, save_videos=True)

    '''ORIGINAL CODE'''
    # ret = main(sys.argv)
    # if ret is not None:
    #     sys.exit(ret)

