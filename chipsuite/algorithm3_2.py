#!/usr/bin/env python3
import cv2
import numpy as np

from chipsuite.algorithm3 import Algorithm3
from chipsuite.powerline import PowerLineDetector


class Algorithm3_2(Algorithm3):
    """
    Algorithm 3.2:
    - template matching with original images and slight blur, sweeping through the image
    - (inherit settable correlation value and posinv name from Algorithm3)
    """
    def __init__(self, *args, **kwargs):
        self.cell_identifier = None
        self.ignore_fillers = True
        self.ignore_cells = False
        super().__init__(*args, **kwargs)


    def set_cell_identifier(self, cell_identifier):
        self.cell_identifier = cell_identifier


    def get_missing_cells(self):
        return self.cells - set(self.cell_templates.keys())


    def prepare(self):
        self.cell_templates = {}
        self.fillers = set(self.posinv_name(b[2]) for b in self.b.bboxes if b[1])
        self.cells = set(self.posinv_name(b[2]) for b in self.b.bboxes if not b[1])
        self.min_corrs = []


    @staticmethod
    def image_preprocess(img):
        #ret, imgThresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        #imgErode = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        #return imgErode
        return cv2.GaussianBlur(img, (5, 5), 0)

    @staticmethod
    def image_template_pad(img):
        sz = 150
        return np.pad(img, [(sz, sz), (sz, sz)], mode='constant', constant_values=127)


    @staticmethod
    def template(cell_image, template_image, mode=cv2.TM_SQDIFF_NORMED, quiet=False):
        #cvhelper.imshowx("cell_image", cell_image, 0, 0)
        #cvhelper.imshowx("template_image", template_image, 1000, 0)
        res = cv2.matchTemplate(cell_image, template_image, mode)
        min_val, _, min_pos, _ = cv2.minMaxLoc(res)
        # fallback in case the value is 1.0
        """if min_val == 1.0:
            # both all black = same
            if not cell_image.any() and not template_image.any():
                min_val = 0.0
            else:
                # use CCORR matching
                res = cv2.matchTemplate(cell_image, template_image, cv2.TM_CCORR_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                min_val = 1.0 - max_val"""
        if not quiet:
            print("template correlation:", min_val, mode==cv2.TM_SQDIFF_NORMED)
        return min_val, min_pos, res


    def decide_corr(self, c):
        return c > self.correlation_value


    def correlate(self, img, acx1, acx2, preprocess=True, quiet=False):
        # naming convention in this function is in the X direction, but it will work the same in Y
        if img.shape[1 if self.p.direction == PowerLineDetector.DIR_X else 0] - acx1 - acx2 < 1:
            # cannot correlate, because we are too small
            return 1.0, False
        if self.p.direction == PowerLineDetector.DIR_X:
            re_croped = img[:,acx1:img.shape[1]-acx2]
        else:
            re_croped = img[acx1:img.shape[0]-acx2,:]
        if preprocess:
            re_croped = self.image_preprocess(re_croped)
        self.cell_image_processed = re_croped
        corr, _, self.matchresult = self.template(self.template_image_processed, re_croped, quiet=quiet)
        corr = corr // 0.001 / 1000
        return corr, not self.decide_corr(corr)


    def sweep_correlate(self, croped, template, preprocess=True, quiet=False):
        # naming convention in this function is in the X direction, but it will work the same in Y
        acx1 = 0
        acx_stepsize = 10
        acy = 0
        if self.p.direction == PowerLineDetector.DIR_X:
            acx2 = -min(self.cx1, 0) - min(self.cx2, 0)
            #window_width = math.ceil((max(croped.shape[1], template.shape[1]) + min(0, self.cx1) + min(0, self.cx2)) * 0.75)
            re_croped = croped[-min(0,self.cy1)+acy:croped.shape[0]+min(0,self.cy2)-acy,:]
        else:
            acx2 = -min(self.cy1, 0) - min(self.cy2, 0)
            re_croped = croped[:,-min(0,self.cx1)+acy:croped.shape[0]+min(0,self.cx2)-acy]
        # slide the window
        self.template_image_processed = self.image_template_pad(self.image_preprocess(template) if preprocess else template)

        while acx2 >= 0:
            yield self.correlate(re_croped, acx1, acx2, preprocess, quiet)
            acx1 += acx_stepsize
            acx2 -= acx_stepsize


    def identify_cell(self, croped, pn, output_images):
        if not self.cell_identifier:
            return
        best_match, corr = self.cell_identifier.identify_cell(croped, pn)
        if best_match:
            print(f"Identified Cell as {best_match} with correlation {corr}")
            output_images["Identified Cell"] = self.cell_templates[best_match]
        else:
            print(f"Could not identify the original cell (we got {len(self.cell_templates)} templates)")


    def analyze_bbox(self, croped, identifier, preprocess=True, identify_cell=True):
        pn = self.posinv_name(identifier)
        if (self.ignore_fillers and pn in self.fillers) or (self.ignore_cells and pn not in self.fillers):
            return False, None # usually too many false positives when filler cells are considered in this algorithm
        if pn not in self.cell_templates:
            # TODO if size is original, store and continue
            #if len(croped) == bh and len(croped[0]) == self.b.bw:
            self.cell_templates[pn] = croped
            print(f"template of {pn} added... ({len(self.get_missing_cells())} missing)")

        # correlate with existing template
        print(f"correlate {identifier}...")

        template = self.cell_templates[pn]
        min_corr = 1.0
        for corr, fin in self.sweep_correlate(croped, template, preprocess):
            if corr < min_corr:
                min_corr = corr
            if fin:
                self.min_corrs.append(min_corr)
                is_diff = False
                break
        else:
            self.min_corrs.append(min_corr)
            #self.cell_templates[pn] = croped #TEST try to not override the template, and find good templates from beginning

            is_diff = True

        output_images = {"Template": self.cell_templates[pn], "Processed Template": self.template_image_processed}
        if preprocess:
            output_images["Processed Cell"] = self.cell_image_processed

        if is_diff and identify_cell:
            self.identify_cell(croped, pn, output_images)

        return is_diff, output_images