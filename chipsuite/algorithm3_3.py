#!/usr/bin/env python3
import cv2
import numpy as np

from chipsuite.algorithm3_2 import Algorithm3_2
from chipsuite.algorithm2 import Algorithm2
from chipsuite import cvhelper


class Algorithm3_3(Algorithm3_2, Algorithm2):
    """
    Algorithm 3.3:
    - use the via detection algorithm from algorithm 2
    - build a via masked based on detected positions by stamping gradients there
    - template matching from algorithm 3.2 without blur
    """
    def __init__(self, *args, **kwargs):
        self.bsz = 40
        super().__init__(*args, **kwargs)
    
    def set_bsz(self, bsz):
        self.bsz = bsz

    def generate_via_mask(self, croped):
        # generate artificial via mask out of detected vias
        via_img, _, vias = self.via_images(croped)
        via_mask = np.full(croped.shape, 127, np.uint8) #np.zeros(croped.shape, np.uint8)
        single_via = cvhelper.ball_template(self.bsz)
        for x, y in vias:
            vx1 = round(x - self.bsz / 2)
            vy1 = round(y - self.bsz / 2)
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
            if vx1 + self.bsz <= croped.shape[1]:
                x2 = x1 + self.bsz - bx1
                bx2 = self.bsz
            else:
                x2 = croped.shape[1] - bx1
                bx2 = self.bsz - ((vx1 + self.bsz) - croped.shape[1])
            if vy1 + self.bsz <= croped.shape[0]:
                y2 = y1 + self.bsz - by1
                by2 = self.bsz
            else:
                y2 = croped.shape[0] - by1
                by2 = self.bsz - ((vy1 + self.bsz) - croped.shape[0])
            try:
                via_mask[y1:y2,x1:x2] = cv2.add(via_mask[y1:y2,x1:x2], single_via[by1:by2,bx1:bx2])
            except:
                print("error, check if this works!")
                print("croped.shape:",via_mask.shape)
                print("vx1,vy1:", vx1, vy1, "bsz:", self.bsz)
                print("y1:y2,x1:x2:", y1, y2, x1, x2, "by1,by2:bx1,bx2:",by1,by2,bx1,bx2)
        return via_img, via_mask


    def analyze_bbox(self, croped, identifier, identify_cell=True):
        via_image, via_mask = self.generate_via_mask(croped)
        is_diff, via_output_images = super().analyze_bbox(via_mask, identifier, False, identify_cell)
        if is_diff:
            return True, {"Via Detection": via_image, "Via Processed Cell": via_mask} | {"Via " + k: v for k, v in via_output_images.items()}
        else:
            return False, None