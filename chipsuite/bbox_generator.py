#!/usr/bin/env python3
import os

import cv2
import numpy as np

from chipsuite.config import Config
from chipsuite import cvhelper
from chipsuite.stitching_info import StitchingInfo


class BboxGenerator:
    def __init__(self, bboxes, stitching_info: StitchingInfo):
        # initialize default correction functions
        self.corr_x = lambda x, y, px: px
        self.corr_y = lambda x, y, py: py

        # set required bboxes
        self.bboxes = bboxes

        # set required stitching info
        self.s = stitching_info


    @staticmethod
    def corr_table(x, y, p, table):
        # Option X, we have a tile number not divisible by 5 and are at the right most edge
        if (x + 4) // 5 == len(table[y//5]) - 1:
            tx = (x + 4) // 5
            # Option XY, we are also on the bottom most edge
            if (y + 4) // 5 == len(table) - 1:
                ty = (y + 4) // 5
                return p - table[ty][tx]
            # Option X1
            if y % 5 == 0:
                return p - table[y//5][tx]
            # Option X2
            v = table[y//5][tx] * (1-y%5/5)
            if y % 5 != 0 and y//5 < len(table)-1:
                v += table[y//5+1][tx] * (y%5/5)
            return p - round(v)

        # Option Y, we have a tile number not divisible by 5 and are at the bottom most edge
        if (y + 4) // 5 == len(table) - 1:
            ty = (y + 4) // 5
            # Option Y1
            if x % 5 == 0:
                return p - table[ty][x//5]
            # Option Y2
            v = table[ty][x//5] * (1-x%5/5)
            if x//5 < len(table[0])-1:
                v += table[ty][x//5+1] * (x%5/5)
            return p - round(v)


        # Option 1, we have the right value in the table
        if x % 5 == 0 and y % 5 == 0:
            return p - table[y//5][x//5]

        # Option 2, we have to interpolate between all surrounding values (max. 4)
        # take the best matching top-left most entry from the table
        v = table[y//5][x//5] * (1-x%5/5) * (1-y%5/5)
        # is there one to the right?
        if x//5 < len(table[0])-1:
            v += table[y//5][x//5+1] * (x%5/5) * (1-y%5/5)
        # are we on an existing line or is there no one below? then we are done, else:
        if y % 5 != 0 and y//5 < len(table)-1:
            # take the one below
            v += table[y//5+1][x//5] * (1-x%5/5) * (y%5/5)
            # is there one to the right?
            if x//5 < len(table[0])-1:
                v += table[y//5+1][x//5+1] * (x%5/5) * (y%5/5)
        return p - round(v)


    def set_corr_x(self, corr_x):
        """
        Parameters:
            corr_x (function): function taking tile X and Y as well as to be
                corrected X coordinate and returning the corrected X coordinate
        
        """
        self.corr_x = corr_x


    def set_corr_y(self, corr_y):
        """
        Parameters:
            corr_y (function): function taking tile X and Y as well as to be
                corrected Y coordinate and returning the corrected Y coordinate
        
        """
        self.corr_y = corr_y


    def correct_stitching(self, element):
        """
        Correct the tile stitching coordinates.

        User-customizable function returning corrected global grid positions of
        the tile provided in the parameter.
        
        Parameters:
            element (dict): Input tile data in MIST stitching format
        
        Returns:
            (int, int): corrected X and Y coordinates of the tile
        
        """
        px, py = element["position"]
        # for 11 <= y <= 13 the stitching is broken (probably due to debris on the images), so correct the stitching here
        px = self.corr_x(element["grid"][0], element["grid"][1], px)
        # correct py here, as stitching was a bit imperfect in the vertical middle (move a bit to the up the more it comes to the center)
        py = self.corr_y(element["grid"][0], element["grid"][1], py)
        return px, py


    def find_bboxes_converted(self, element, also_standard_cells=False):
        """
        Generator to find bounding boxes inside a tile.
        
        Returns a generator that finds every bounding box that matches to the
        stitching tile provided in the parameter. For each bounding box on the
        tile, it yields a tuple of coordinates on this tile, width, height all
        calculated referenced to the current tile. The fourth return value is the
        bounding box element itself as it provides further information.
        
        Parameters:
            element (dict): Input tile data in MIST stitching format
            also_standard_cells (bool): Also yield standard cells. Defaults to
                False, only yielding filler cell related bounding boxes.
        
        Yields:
            ((int, int), int, int, ((float, float, float, float), bool, string):
                A tuple consisting of the position of the bounding box, adjusted
                to the reference of the tile, width, height of the bounding box,
                a tuple of the original bbox element, containing a tuple with the
                source coordinates, a bool deciding that it is a filler cell and
                a descriptive string of the GDS representation of this bbox.
        
        """
        px, py = self.correct_stitching(element)
        for bbox in self.bboxes:
            # we are only interested in filler cells
            if not bbox[1] and not also_standard_cells:
                continue
            # find the boxes that are inside px,py,px+pw,py+ph
            bx1 = bbox[4][0]
            by1 = bbox[4][1]
            bx2 = bbox[4][2]
            by2 = bbox[4][3]
            if (bx2 < px or
                by2 < py or
                bx1 >= px+self.s.pw or
                by1 >= py+self.s.ph):
                continue
            # fine-rotate boxes that can be shown
            bw = round(bx2 - bx1)
            bh = round(by2 - by1)
            bxy1 = (round(bx1 - px), round(by1 - py))
            yield bxy1, bw, bh, bbox



    def draw_boxes(self, x, y, fract=1, highlight=[], hide_boxes=False):
        element = self.s.stitching[x, y]
        image = cv2.imread(os.path.join(self.s.image_folder, element["file"]))
        if not hide_boxes:
            for bxy1, bw, bh, bbox in self.find_bboxes_converted(element, True):
                color = Config.COLOR_FILLER if bbox[1] else Config.COLOR_NORMAL # has no effect currently, to enable black rectangles (standardcells) for debugging, add ,True to the find_bboxes_converted call.
                #image = cv2.rectangle(image, bxy1, (bxy1[0]+bw, bxy1[1]+bh), color, 10)
                points = []
                px, py = self.correct_stitching(element)
                for point in bbox[3]:
                    points.append([round(point[0] - px), round(point[1] - py)])
                image = cv2.polylines(image, [np.array(points)], True, color, 10)
        
        if highlight:
            for c in highlight:
                image = cv2.rectangle(image, (c[0], c[1]), (c[2], c[3]), Config.COLOR_HIGHLIGHT, 10)
        image = cv2.resize(image, (self.s.pw//fract, self.s.ph//fract))
        return image


    def show_boxes(self, x, y, fract=1, highlight=[], hide_boxes=False):
        cvhelper.imshowy("Boxes", self.draw_boxes(x, y, fract, highlight, hide_boxes))
    
    def xyshow(self, tx, ty, qx, qy):
        cv2.namedWindow("Boxes", cv2.WINDOW_NORMAL)
        D_LIST = [1, 10]
        d = 1
        while True:
            self.set_corr_x(lambda x, y, px: px - qx)
            self.set_corr_y(lambda x, y, py: py - qy)
            cv2.imshow("Boxes", self.draw_boxes(tx, ty))
            key = cv2.waitKey(0)
            if key == 83: # RIGHT
                qx += D_LIST[d]
            elif key == 81: # LEFT
                qx -= D_LIST[d]
            elif key == 84: # DOWN
                qy += D_LIST[d]
            elif key == 82: # UP
                qy -= D_LIST[d]
            elif key == 225: # SHIFT
                d += 1
                if d >= len(D_LIST):
                    d = 0
            else:
                break
        cv2.destroyAllWindows()
        print(f"X: {qx} Y: {qy}")