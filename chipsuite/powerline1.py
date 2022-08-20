#!/usr/bin/env python3
import cv2
import math
import numpy as np

from chipsuite import cvhelper
from chipsuite.algorithm2 import Algorithm2
from chipsuite.bbox_generator import BboxGenerator
from chipsuite.powerline import PowerLineDetector


class PowerLineDetector1(PowerLineDetector):
    def __init__(self, *args, **kwargs):
        self.continuity = 0.9

        self.blur_value_x = 11
        self.blur_value_y = 11
        self.blur_halved = True
        self.adaptive_gaussian_threshold = 11
        self.adaptive_gaussian_const = -4
        self.erode_iter = 2
        self.dilate_iter = 3
        self.min_via_radius = 2
        self.max_via_radius = 10
        self.min_variance = 0
        self.min_correlation = 1
        super().__init__(*args, **kwargs)


    def set_values(self, first_iterations, iterations, threshold, continuity, hough_threshold_factor):
        self.continuity = continuity


    def set_blur_values(self, blur_value_x, blur_value_y, blur_halved=False):
        self.blur_value_x = blur_value_x
        self.blur_value_y = blur_value_y
        self.blur_halved = blur_halved
    
    
    def set_adaptive_threshold_values(self, adaptive_gaussian_threshold, adaptive_gaussian_const):
        self.adaptive_gaussian_threshold = adaptive_gaussian_threshold
        self.adaptive_gaussian_const = adaptive_gaussian_const
    
    
    def set_erode_dilate_values(self, erode_iter, dilate_iter):
        self.erode_iter = erode_iter
        self.dilate_iter = dilate_iter
    
    
    def set_via_values(self, min_via_radius, max_via_radius, min_variance, min_correlation):
        self.min_via_radius = min_via_radius
        self.max_via_radius = max_via_radius
        self.min_variance = min_variance
        self.min_correlation = min_correlation


    def naive_get_viamask(self, img, boxdim):
        alg2 = Algorithm2(BboxGenerator([], None))
        alg2.set_via_values(self.min_via_radius, self.max_via_radius, self.min_variance, self.min_correlation, 0)
        alg2.set_blur_values(self.blur_value_x, self.blur_value_y, self.blur_halved)
        alg2.set_adaptive_threshold_values(self.adaptive_gaussian_threshold, self.adaptive_gaussian_const)
        alg2.set_erode_dilate_values(self.erode_iter, self.dilate_iter)
        _, _, vias = alg2.via_images(img)
        via_mask = np.zeros(img.shape, np.uint8)
        single_box = np.full(boxdim, 255, np.uint8)
        for x, y in vias:
            vx1 = round(x - boxdim[1] / 2)
            vy1 = round(y - boxdim[0] / 2)
            if vx1 >= 0:
                x1 = vx1
                bx1 = 0
            else:
                x1 = 0
                bx1 = -vx1
            if vy1 >= 0:
                y1 = vy1
                by1 = 0
            else:
                y1 = 0
                by1 = -vy1
            if vx1 + boxdim[1] <= img.shape[1]:
                x2 = x1 + boxdim[1] - bx1
                bx2 = boxdim[1]
            else:
                x2 = img.shape[1]
                bx2 = boxdim[1] - ((vx1 + boxdim[1]) - img.shape[1])
            if vy1 + boxdim[0] <= img.shape[0]:
                y2 = y1 + boxdim[0] - by1
                by2 = boxdim[0]
            else:
                y2 = img.shape[0]
                by2 = boxdim[0] - ((vy1 + boxdim[0]) - img.shape[0])
            via_mask[y1:y2,x1:x2] = cv2.add(via_mask[y1:y2,x1:x2], single_box[by1:by2,bx1:bx2])
        return via_mask


    def detect_powerlines(self, img, stepsize=1, blur_size=11):
        """# create a mask of only the vias
        ret, imgThresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        imgErode = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        """
        imgErode = self.naive_get_viamask(img, (7, 30) if self.direction == PowerLineDetector.DIR_X else (30, 7) if self.direction == PowerLineDetector.DIR_Y else (30, 30))
        #cvhelper.imshowy("Erode", imgErode)

        # detect all the lines
        lines = cv2.HoughLinesP(imgErode, 1, np.pi/180, round(img.shape[1 if self.direction == PowerLineDetector.DIR_X else 0]*0.4), minLineLength=10, maxLineGap=50000)
        #self.filtered_values_show(img, [], lines, "P")

        if self.direction == PowerLineDetector.DIR_AUTO:
            # find dominate line orientation
            angles = []
            anglePrecisionPlaces = 2

            for line in lines:
                x1, y1, x2, y2 = line[0]

                if y1 == y2:
                    angle = 90.0 # lim x->inf arctan(x) = 90 degree
                else:
                    angle = math.atan((x1-x2)/(y1-y2)) * 180 / math.pi

                angles.append(round(angle, anglePrecisionPlaces))

            blur_size_angle = 3

            # calculate the dominate angle by bluring the histogram of all found angles. this can be seen as a moving average
            if all(angle == angles[0] for angle in angles):
                # if all are the same, the blur will fail
                dominantAngle = angles[0]
            else:
                countsAngle, binsAngle = np.histogram(angles, bins=np.arange(min(angles), max(angles)+stepsize, stepsize))
                blurAngle = cv2.blur(countsAngle*blur_size_angle,(1,blur_size_angle)).reshape(countsAngle.shape)
                dominantAngle = binsAngle[np.nanargmax(blurAngle)]

            directedAngle = (dominantAngle + 360) % 180
            if directedAngle > 45 and directedAngle < 135:
                self.direction = PowerLineDetector.DIR_X
            else:
                self.direction = PowerLineDetector.DIR_Y
            print("Detected Direction:", "X" if self.direction == PowerLineDetector.DIR_X else "Y")

        return self.filter_lines(img, lines, "P", stepsize, blur_size)
