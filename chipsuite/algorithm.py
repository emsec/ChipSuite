#!/usr/bin/env python3
import abc
import cv2
from io import StringIO
import __main__
import math
import numpy as np
import os
import sys
import time

from chipsuite.bbox_generator import BboxGenerator
from chipsuite import cvhelper
from chipsuite.powerline import PowerLineDetector
from chipsuite.stitching_info import StitchingInfo


CV_X = 0 # dual monitor on 1920


class Algorithm(metaclass=abc.ABCMeta):
    # modes for fused image
    MODE_DEFAULT = 0
    MODE_FUSE = 1
    MODE_THRESHOLDED = 2
    MODE_DRAW_BOXES = 3
    MODE_FIND_CELLS = 4
    MODE_ONLY_CELLS = 5

    def __init__(self, bbox_generator: BboxGenerator, stitching_info: StitchingInfo = None, powerline_detector: PowerLineDetector = None):
        # initialize default fraction when fusing images (1, 2, 4, 8, ...)
        self.fract = 1

        # initialize default cell crops
        self.cx1 = 0
        self.cy1 = 0
        self.cx2 = 0
        self.cy2 = 0

        # initialize interactive mode
        self.silent = False
        self.timeout = 0
        # list of tupels containing tile X, Y and cell identifier
        self.known_cells = []

        # initialize counters
        self.asked_count = 0
        self.valid_cell_count = 0
        self.tile_count = 0

        # set required bbox generator
        self.b = bbox_generator

        # set bboxes on the edge of a tile?
        self.bboxes_on_edge = True

        # set powerline detector if any. If None, do not correct coordinates
        self.p = powerline_detector

        # set required stitching info
        if not stitching_info:
            self.s = self.b.s
        else:
            self.s = stitching_info

        # initialize output folder
        self.output_folder = ""

        # set HTML output to default off to be able to interactively debug algorithms
        self.html_filename = None

        # special mode for getting all the correlation values
        self.special_mode = False


    def set_fract(self, fract):
        self.fract = fract


    def set_interactive_mode(self, interactive, timeout, known_cells=[]):
        self.silent = not interactive
        self.timeout = timeout
        self.known_cells = known_cells
    
    
    def set_output_folder(self, output_folder):
        self.output_folder = output_folder


    def set_cell_crop(self, cx1, cy1, cx2, cy2):
        self.cx1 = cx1
        self.cy1 = cy1
        self.cx2 = cx2
        self.cy2 = cy2
    
    def set_html_output(self, filename):
        self.html_filename = filename


    def cut_tile_edges(self, element, bboxes_list):
        min_x, min_y = (min(w) for w in zip(*list(z[0] for z in bboxes_list)))
        max_x, max_y = (max(w) for w in zip(*list(z[0] + np.array(z[1:3]) for z in bboxes_list)))
        min_x = round(min_x) if min_x > 0 else 0 # clip left to 0
        max_x = round(max_x) if max_x < self.s.pw - 1 else self.s.pw - 1 # clip right to width - 1
        min_y = round(min_y) if min_y > 0 else 0 # clip top to 0
        max_y = round(max_y) if max_y < self.s.ph - 1 else self.s.ph - 1 # clip bottom to height - 1
        #print("cut ({},{}) to ({},{})".format(min_x, min_y, max_x, max_y))
        return min_x, min_y, max_x, max_y


    def is_mode_requiring_a_canvas(self):
        return not (self.mode == Algorithm.MODE_DEFAULT or self.mode == Algorithm.MODE_FIND_CELLS or self.mode >= Algorithm.MODE_ONLY_CELLS)


    def analyze(self, mode=MODE_DEFAULT, starttime="not set", start=None):
        self.mode = mode
        if self.is_mode_requiring_a_canvas():
            dimensions = (math.ceil((self.s.st_height+self.s.ph)/self.fract), math.ceil((self.s.st_width+self.s.pw)/self.fract))
            self.canvas = np.zeros((dimensions + (3,)) if mode == Algorithm.MODE_DRAW_BOXES else dimensions, dtype="uint8")
        if self.html_filename is not None:
            self.html_file = open(os.path.join(self.output_folder, self.html_filename), "w")
            self.html_file.write(f"<!DOCTYPE html><html><head><meta charset=\"utf-8\" /><title>ChipSuite Results of {os.path.basename(__main__.__file__)}</title><style>h2:not(:first-of-type){{page-break-before:always}}</style></head><body><p>Start Time: {starttime}</p>\n")
        if start is None:
            self.analyze_loop()
        else:
            self.analyze_loop(start[0], start[1])
            self.analyze_loop(0, 0, stop=start)
        if self.html_filename is None:
            print(f"askedCount: {self.asked_count}")
            print(f"validCellCount: {self.valid_cell_count}")
            print(f"tileCount: {self.tile_count}")
        else:
            self.html_file.write(f"<h2>Summary</h2><table><tr><td>askedCount:</td><td>{self.asked_count}</td></tr><tr><td>validCellCount:</td><td>{self.valid_cell_count}</td></tr><tr><td>tileCount:</td><td>{self.tile_count}</td></tr></table>\n")
            self.html_file.write(f"<p>End Time: {time.strftime('%c')}</p>")
            self.html_file.write("</body></html>\n")
            self.html_file.close()
        if self.is_mode_requiring_a_canvas():
            cv2.imwrite(os.path.join(self.output_folder, "stitched.png"), self.canvas)
            cvhelper.imshowy("Stitched Fuse", self.canvas)


    def analyze_loop(self, sx=0, sy=0, mode=None, stop=None):
        if mode is not None:
            self.mode = mode
        for y in range(sy, self.s.t_height):
            for x in range(sx, self.s.t_width):
                if stop == (x, y) or (x, y) not in self.s.stitching:
                    return
                if not self.special_mode:
                    print("Tile:", x, y)
                abort, valid_cells_on_tile = self.analyze_tile(x, y)
                self.valid_cell_count += valid_cells_on_tile
                if abort:
                    break
                if self.mode != Algorithm.MODE_FIND_CELLS and self.mode != Algorithm.MODE_ONLY_CELLS:
                    self.tile_count += 1
            else:
                sx = 0
                continue
            break


    def analyze_tile(self, x, y, mode=None, filter=None):
        if mode is not None:
            self.mode = mode
        i = 0
        valid_cells = 0
        element = self.s.stitching[x, y]

        if self.mode == Algorithm.MODE_DRAW_BOXES:
            image = self.b.draw_boxes(x, y, self.fract)
        else:
            image = cv2.imread(os.path.join(self.s.image_folder, element["file"]), cv2.IMREAD_GRAYSCALE)
        
        if self.is_mode_requiring_a_canvas():
            if self.mode == Algorithm.MODE_THRESHOLDED:
                if not hasattr(self, 'threshold_value'):
                    raise NotImplementedError("Thresholded fusing is only possible with an algorithm that supports thresholds, because we need a threshold value.")
                thresholded = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_BINARY)[1]
                image = thresholded
            px, py = self.b.correct_stitching(element)
            self.canvas = cvhelper.cvStampImageScaled(self.canvas, image, int(px//self.fract), int(py//self.fract), int(self.s.pw//self.fract), int(self.s.ph//self.fract))
            return False, 0
        
        # crop out image
        bboxes_list = list(self.b.find_bboxes_converted(element, self.mode >= Algorithm.MODE_FIND_CELLS)) # TODO was on MODE_ONLY_CELLS which kinda makes not much sense only if algo 3 is used and only fillers desired
        if not bboxes_list:
            # if there is no bounding box on this tile (border tiles e.g.) we do not need to analyze this tile further
            return False, 0
        cx, cy, max_x, max_y = self.cut_tile_edges(element, bboxes_list)
        image_croped = image[cy:max_y, cx:max_x]
        cw = max_x - cx
        ch = max_y - cy

        if self.p is not None:
            # do the powerline detection
            power_lines = self.p.detect_powerlines(image_croped)

        for bxy1, bw, bh, bbox in bboxes_list:
            if filter is not None and filter not in bbox[2]:
                continue

            # make width a bit thinner to cope for via lines (power supply in our design)
            bxy2 = (bxy1[0] + bw, bxy1[1] + bh)
            bw -= self.cx1 + self.cx2
            bh -= self.cy1 + self.cy2
            bx = bxy1[0] + self.cx1
            by = bxy1[1] + self.cy1

            if not self.bboxes_on_edge and (min(image_croped.shape[0],max(0,by-cy+bh))-max(0,by-cy) < bh or min(image_croped.shape[1],max(0,bx-cx+bw))-max(0,bx-cx) < bw):
                continue # in this mode we are not interested at all for cells touching the corner

            # use power_lines now!
            if self.p is not None and power_lines:
                # correct crop against detected power lines
                if self.p.direction == PowerLineDetector.DIR_X:
                    center_value = by + bh/2 - cy
                else:
                    center_value = bx + bw/2 - cx
                for j, value in enumerate(power_lines[1:]):
                    if center_value >= power_lines[j] and center_value < value:
                        break
                else:
                    # it seems that the correct range is beyond the image edges
                    continue
                if self.p.direction == PowerLineDetector.DIR_X:
                    by = power_lines[j] + cy + self.cy1
                    bh = power_lines[j + 1] - power_lines[j] - (self.cy1 + self.cy2)
                else:
                    bx = power_lines[j] + cx + self.cx1
                    bw = power_lines[j + 1] - power_lines[j] - (self.cx1 + self.cx2)

            if bw <= 0 and bh <= 0:
                continue
            croped = image_croped[max(0,by-cy):max(0,by-cy+bh), max(0,bx-cx):max(0,bx-cx+bw)].copy()
            if croped.shape[0] / bh < 0.5 or croped.shape[1] / bw < 0.5:
                continue # we only have less than half the box in our current tile, often false positives...
            if self.html_filename is not None or self.special_mode:
                sys.stdout = StringIO()
            candidate, output_images = self.analyze_bbox(croped, bbox[2])
            if self.html_filename is not None or self.special_mode:
                algorithm_output = sys.stdout.getvalue()
                sys.stdout = sys.__stdout__
            if not candidate:
                continue
            if self.special_mode:
                lowest_match = 2.0 # this is an error condition
                tmp1 = algorithm_output.split("\ncorrelate")
                if len(tmp1) > 1:
                    tmp2 = tmp1[1].split("template correlation: ")
                    if len(tmp2) > 1:
                        lowest_match = min(float(x.split()[0]) for x in tmp2[1:])
                print(f"{x}_{y}_{i} {lowest_match:0.4f} {bbox[2]}")
                i += 1
                continue
            output_images = {"Cell Cutout": croped} | output_images
            if self.mode == Algorithm.MODE_FIND_CELLS:
                return True, 0
            self.asked_count += 1
            if self.html_filename is None:
                print(bxy1[0], bxy1[1], bw, bh)
                print(croped.shape[1], croped.shape[0])
                print(bbox[2])
                print(f"askedCount: {self.asked_count}")
                if not self.silent:
                    cv_y = 0
                    for key, image in output_images.items():
                        if image is not None and len(image):
                            cvhelper.imshowx(key, image, CV_X, cv_y)
                            cv_y += 400
                    key = cv2.waitKey(self.timeout) & 0xFF
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        return True, i
                    elif key == ord("s"):
                        if "Output" in output_images and output_images["Output"] is not None and len(output_images["Output"]):
                            cv2.imwrite(os.path.join(self.output_folder, f"conflict_{x}_{y}_{i}_detection.png"), output_images["Output"])
                        boxes = self.b.draw_boxes(x, y, self.fract, [list(bxy1) + list(bxy2)])
                        cv2.imwrite(os.path.join(self.output_folder, f"conflict_{x}_{y}_{i}.png"), boxes)
                        cv2.imshow("Boxes", boxes)
                        cv2.waitKey(0)
                    cv2.destroyAllWindows()
                if (x, y, bbox[2]) in self.known_cells:
                    print("VALID")
                    valid_cells += 1
            else:
                f = f"{self.html_filename}_images"
                c = f"{x}_{y}_{i}"
                rf = os.path.join(self.output_folder, f)
                if not os.path.exists(rf):
                    os.makedirs(rf)
                im = os.path.join(f, c)
                self.html_file.write(f"<h2>Candidate {c}</h2><table><tr><td>Cell:</td><td>{bbox[2]}</td></tr><tr><td>X,Y,W,H:</td><td>{bxy1[0]} {bxy1[1]} {bw} {bh}</td></tr><tr><td>Shape:</td><td>{croped.shape[1]} {croped.shape[0]}</td></tr><tr><td>Algorithm:</td><td><pre>{algorithm_output}</pre></td></tr>")
                j = 0
                for key, image in output_images.items():
                    if image is not None and len(image):
                        self.html_file.write(f"<tr><td>{key}:</td><td><img src=\"{im}_{j}.png\" alt=\"\" /></td></tr>")
                        cv2.imwrite(os.path.join(rf, f"{c}_{j}.png"), image)
                        j += 1
                if (x, y, bbox[2]) in self.known_cells:
                    print("VALID")
                    self.html_file.write(f"<tr><th colspan=\"2\">VALID</th></tr>")
                    valid_cells += 1
                self.html_file.write("</table>\n")
                self.html_file.flush()
            i += 1
        return False, valid_cells


    @abc.abstractmethod
    def analyze_bbox(self, croped, identifier):
        """
        returns:
        - candidate (bool)
          in case the analyzed bbox was found to be a candidate that has to be taken into consideration
        - output_images (dict of cv2/numpy-images)
          images that help in distinguishing between true and false positive and give further information
          will be ignored when candidate equals False
        """

        raise NotImplementedError