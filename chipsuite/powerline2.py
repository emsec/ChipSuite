#!/usr/bin/env python3
import cv2
import math
import numpy as np

from chipsuite import cvhelper
from chipsuite.powerline import PowerLineDetector


class PowerLineDetector2(PowerLineDetector):
    def __init__(self, *args, **kwargs):
        self.first_iterations = 1
        self.iterations = 4 # default to 1?
        self.threshold = 120
        self.continuity = 0.9
        self.hough_threshold_factor = 0.7
        super().__init__(*args, **kwargs)


    def set_values(self, first_iterations, iterations, threshold, continuity, hough_threshold_factor):
        self.first_iterations = first_iterations
        self.iterations = iterations
        self.threshold = threshold
        self.continuity = continuity
        self.hough_threshold_factor = hough_threshold_factor


    @staticmethod
    def hough_edges(img, direction, iterations=1, extended_method=True, threshold=120, hough_threshold_factor=0.7):
        if direction == PowerLineDetector.DIR_X:
            if img.shape[1] < img.shape[0]:
                img = np.concatenate((img, img, img, img), axis=1)
            imgBlured = cv2.blur(img, (51, 5))
            imgBluredEdges = cv2.Sobel(imgBlured, -1, 0, 1, 5)
            dimension = img.shape[1]
        elif direction == PowerLineDetector.DIR_Y:
            if img.shape[0] < img.shape[1]:
                img = np.concatenate((img, img, img, img), axis=0)
            imgBlured = cv2.blur(img, (5, 51))
            imgBluredEdges = cv2.Sobel(imgBlured, -1, 1, 0, 5)
            dimension = img.shape[0]
        else:
            assert False, "Unsupported Direction"
        #cvhelper.imshowy("outputOriginal", imgBluredEdges)
        imgBluredEdgesErode = cv2.erode(imgBluredEdges, None, iterations=iterations)
        if extended_method:
            imgErode = np.zeros(img.shape, dtype="uint8")
            cv2.normalize(imgBluredEdgesErode, imgErode, 0, 255, cv2.NORM_MINMAX)
            imgErode2 = cv2.threshold(imgErode, threshold, 255, cv2.THRESH_BINARY)[1]
            imgErode3 = cv2.blur(imgErode2, (101, 5) if direction == PowerLineDetector.DIR_X else (5, 101))
            imgErode = imgErode3 * 255 # hard threshold, everything that is brighter than 0 should be capped to 255
        else:
            imgErode = imgBluredEdgesErode * 255
        #cvhelper.imshowy("outputOriginal", imgErode)

        # accumulator threshold most likely to be in the range of half of the image dimensions
        #lines = cv2.HoughLines(imgErode, 1, np.pi/180, round(dimension * 0.71))
        #hough_threshold = round(dimension * (0.71 + (4-iterations) * 0.008)) # HACK, somehow otherwise tile 40nm/57/0 doesn't work nice
        hough_threshold = round(dimension * hough_threshold_factor)
        angle_threshold = 1
        if direction == PowerLineDetector.DIR_X:
            lines = cv2.HoughLines(imgErode, 1, np.pi/180, hough_threshold, min_theta=0, max_theta=math.pi/2+angle_threshold) # TODO possible bug with min_theta, should be math.pi/2-angle_threshold: https://github.com/opencv/opencv/issues/21983
        else:
            lines1 = cv2.HoughLines(imgErode, 1, np.pi/180, hough_threshold, min_theta=math.pi-angle_threshold, max_theta=math.pi)
            lines2 = cv2.HoughLines(imgErode, 1, np.pi/180, hough_threshold, min_theta=0, max_theta=angle_threshold)
            if lines1 is None:
                lines = lines2
            elif lines2 is None:
                lines = lines1
            else:
                lines = np.append(lines1, lines2, axis=0)
        if lines is None:
            lines = cv2.HoughLines(imgErode, 1, np.pi/180, hough_threshold)
        #print("lines:", lines)
        
        if lines is not None:
            imgHough = PowerLineDetector.hough_lines_show(img, lines, "")
        else:
            imgHough = img
        #cvhelper.imshowy("outputOriginal", imgHough)
        return lines, imgErode

    def detect_powerlines(self, img, stepsize=1, blur_size=11, extended_method=False):
        if self.direction == PowerLineDetector.DIR_AUTO:
            # try both directions and choose which is more likely
            lines_x, imgErode_x = PowerLineDetector2.hough_edges(img, PowerLineDetector.DIR_X, self.first_iterations, extended_method, self.threshold, self.hough_threshold_factor)
            lines_y, imgErode_y = PowerLineDetector2.hough_edges(img, PowerLineDetector.DIR_Y, self.first_iterations, extended_method, self.threshold, self.hough_threshold_factor)
            if lines_x is not None and (lines_y is None or len(lines_x) > len(lines_y)):
                self.direction = PowerLineDetector.DIR_X
                lines = lines_x
                imgErode = imgErode_x
            elif lines_y is not None:
                self.direction = PowerLineDetector.DIR_Y
                lines = lines_y
                imgErode = imgErode_y
            else:
                self.direction = PowerLineDetector.DIR_UNDETECTED
                lines = None
            if self.direction == PowerLineDetector.DIR_X or self.direction == PowerLineDetector.DIR_Y:
                print("Detected Direction:", "X" if self.direction == PowerLineDetector.DIR_X else "Y")
        lines, imgErode = PowerLineDetector2.hough_edges(img, self.direction, self.iterations, True, self.threshold, self.hough_threshold_factor)
        hough_mode = ""

        return self.filter_lines(img, lines, hough_mode, stepsize, blur_size)