#!/usr/bin/env python3
import abc
import cv2
import math
import numpy as np

from chipsuite import cvhelper

class PowerLineDetector(metaclass=abc.ABCMeta):
    # directions for tile orientation detection
    DIR_Y = 0
    DIR_X = 1
    DIR_AUTO = 2
    DIR_UNDETECTED = 3

    def __init__(self):
        # initialize chip wide tile orientation
        self.direction = PowerLineDetector.DIR_AUTO
        self.averages_height = None
        self.averages_height_count = 0


    @staticmethod
    def hough_line_convert(line, mode="P", shape=(0,0)):
        if mode == "P":
            return line
        else:
            rho, theta = line.ravel()
            a = math.cos(theta)
            b = math.sin(theta)
            x1 = a * rho
            y1 = b * rho
            off_x = (shape[1]-1) * (-b)
            off_y = (shape[0]-1) * (a)
            if y1 > x1:
                x2 = round(x1 - off_x)
                y2 = round(y1 - off_y)
            else:
                x2 = round(x1 + off_x)
                y2 = round(y1 + off_y)
            return round(x1), round(y1), x2, y2


    @staticmethod
    def hough_lines_show(img, lines, mode="P"):
        imgColor = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = PowerLineDetector.hough_line_convert(line[0], mode, img.shape)
                cv2.line(imgColor, (round(x1), round(y1)), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
        return imgColor

    def filtered_values_show(self, img, values, hough_lines=[], hough_mode=""):
        imgHough = PowerLineDetector.hough_lines_show(img, hough_lines, hough_mode)
        complete_size = img.shape[1 if self.direction == PowerLineDetector.DIR_X else 0]
        for value in values:
            if self.direction == PowerLineDetector.DIR_X:
                cv2.line(imgHough, (0, value), (complete_size, value), (0, 255, 0), 3, cv2.LINE_AA)
            elif self.direction == PowerLineDetector.DIR_Y:
                cv2.line(imgHough, (value, 0), (value, complete_size), (0, 255, 0), 3, cv2.LINE_AA)
        cvhelper.imshowy("Hough Lines", imgHough)


    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    # requires self.continuity
    def filter_lines(self, img, lines, hough_mode="", stepsize=1, blur_size=11):
        if lines is None:
            print("Warning! No Edges detected. Maybe decrease iterations?")
            return []

        # get all the X or Y (depending on the detected or given direction) coordinates of all power lines
        values = []
        for line in lines:
            x1, y1, _, _ = PowerLineDetector.hough_line_convert(line[0], hough_mode, img.shape)
            value = y1 if self.direction == PowerLineDetector.DIR_X else x1
            if value < 0:
                print("Warning, negative coordinates in the Power Line Detection detected.")
                continue
            values.append(value)

        values.append(0)
        size = img.shape[0 if self.direction == PowerLineDetector.DIR_X else 1]
        values.append(size-1)
        values.sort()

        # filter the coordinates to remove possible duplicates, first get the median value
        last_value = 0
        sizes = []
        for value in values[1:]:
            sizes.append(value - last_value)
            last_value = value
        counts, bins = np.histogram(sizes, bins=np.arange(min(sizes), max(sizes) + stepsize, stepsize))
        blur = cv2.blur(counts * blur_size, (1, blur_size)).reshape(counts.shape)
        blur[:round(len(blur)*(1-self.continuity))] = 0
        median_size = bins[np.nanargmax(blur)]
        #print("Median Size:", median_size)

        # strategy now is to correct errors by finding the most adjacent truly detected power lines and then completing missing ones
        # use here the accumulated averaged median value
        if self.averages_height_count < 2:
            average_height = self.averages_height
        else:
            average_height = self.averages_height / self.averages_height_count
        if average_height is None:
            average_height = median_size
        adjacent_matches = 0
        adjacent_dict = {}
        diff = 0
        for i, value in enumerate(values[1:]):
            # i is already decremented by 1 here
            diff += value - values[i]
            # if the difference between the adjacent lines is less than remaining space of the tolerated area, count it as if it was one line
            if diff < average_height * (1 - self.continuity):
                continue
            # if the difference is in range of the tolerance, count it as valid
            if diff > average_height * self.continuity and diff < average_height * (2 - self.continuity):
                adjacent_matches += 1
            else:
                adjacent_dict[i] = adjacent_matches
                adjacent_matches = 0
            diff = 0
        adjacent_dict[len(values) - 1] = adjacent_matches
        #print("adjacent dict:", adjacent_dict)

        # start with the last index of the longest adjacent sequence of already valid
        #print(values)
        index = sorted(adjacent_dict, key=adjacent_dict.get)[-1]
        if index == len(values) - 1 and adjacent_dict[index] > 1: # TODO check if second part is required
            # fix an error when the right/bottommost index would get selected, which is the edge of the tile
            # this bug shouldn't be apparent at the left/topmost index, as the best index is safely > 0
            index -= 1
        value = values[index]
        filtered_values = [value]
        # go up first, then down
        search_dir = -1
        while True:
            expected_value = value + search_dir * average_height
            nearest_value = PowerLineDetector.find_nearest(values, expected_value)
            if abs(nearest_value - expected_value) < average_height * (1 - self.continuity):
                # take the detected exact line
                value = nearest_value
            else:
                # artifically generate a new one based on the averaged median
                value = expected_value
            if value < 0:
                # now go down
                search_dir = 1
                value = filtered_values[0]
            else:
                if value >= size:
                    break
                filtered_values.append(value)

        filtered_values = sorted(filtered_values)

        #print(filtered_values)

        diffs = [value-filtered_values[i] for i, value in enumerate(filtered_values[1:])]
        if not diffs:
            return []
        new_average_height = sum(diffs) / len(diffs)

        powerline_values = [round(value) for value in filtered_values]

        # only for debugging purposes
        #self.filtered_values_show(img, powerline_values, lines, hough_mode)

        if self.averages_height_count < 1:
            self.averages_height = new_average_height
        else:
            self.averages_height += new_average_height
        self.averages_height_count += 1
        print("average of this tile:", new_average_height, "total average:", self.averages_height / self.averages_height_count)

        return powerline_values


    @abc.abstractmethod
    def detect_powerlines(self, img, **kwargs):
        raise NotImplementedError

class FakePowerLineDetector(PowerLineDetector):
    def __init__(self, direction):
        self.direction = direction
    
    def detect_powerlines(self, img, **kwargs):
        return []