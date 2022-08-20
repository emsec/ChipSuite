#!/usr/bin/env python3
import cv2

from chipsuite.algorithm import Algorithm


class Algorithm1(Algorithm):
    def set_threshold_value(self, threshold_value):
        self.threshold_value = threshold_value


    def set_pixel_threshold(self, pixel_threshold):
        self.pixel_threshold = pixel_threshold


    def analyze_bbox(self, croped, identifier):
        thresh = cv2.threshold(croped, self.threshold_value, 255, cv2.THRESH_BINARY)[1]
        mean = cv2.mean(thresh)[0]
        sumElems = cv2.sumElems(thresh)[0] / 255
        if sumElems > self.pixel_threshold: # ignore mean > 20
            print("Mean:", mean)
            print("Sum:", sumElems)
            return True, {"Thresholded": thresh}
        return False, None