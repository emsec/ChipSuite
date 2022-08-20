#!/usr/bin/env python3
import cv2
import numpy as np

from chipsuite.algorithm import Algorithm


class Algorithm3(Algorithm):
    def set_correlation_value(self, correlation_value):
        self.correlation_value = correlation_value


    def get_missing_cells(self):
        return self.fillers - set(self.cell_templates.keys())


    def prepare(self):
        self.cell_templates = {}
        self.fillers = set(self.posinv_name(b[2]) for b in self.b.bboxes) # if b[1])
        # start at a safe location (10,10) should always have a good image quality
        print("FIRST RUN")
        self.analyze_loop(10, 10, Algorithm.MODE_FIND_CELLS)
        if len(self.get_missing_cells()):
            # start from the beginning, hopefully up to the first start, find all remaining
            self.analyze_loop(0, 0, Algorithm.MODE_FIND_CELLS)
            missing_cells = self.get_missing_cells()
            assert len(missing_cells) == 0, f"Still missing cell templates, cannot run. The following are missing: {', '.join(missing_cells)}. Most probably they are cropped too small to be eligible as a template."
        print("FIRST RUN DONE")



    # from a cell identifier, build a position invariant string
    # cell identifiers could look like this:
    # CellReference ("FILL4BWP35P140HVT$1", at (644.5600000000002, 1265.7000000000003), rotation None, magnification None, reflection True)
    #                                  ^        ^- mainly we want to get rid of these coordinates
    #                                  '- no special meaning, some GDSII cells seem to have this, but they are equal to them without "$1"
    # after the posinv_name() call the identifiers look like this:
    # CellReference ("FILL4BWP35P140HVT", rotation None, magnification None, reflection True)
    @staticmethod
    def posinv_name(n):
        return (",".join(n.split(",")[:1]+n.split(",")[3:])).replace("$1", "")


    @staticmethod
    def template(b, a, mode=cv2.TM_CCOEFF_NORMED):
        # first step would be to make the larger of the two so large that both dimensions are about the same size? (or make the original (not the template) even quite a lot bigger so template matching really detects the template there)
        if len(b[0]) > len(a[0]):
            c = b
            b = a
            a = c
        if len(b) > len(a):
            c = np.zeros((len(b), len(a[0])), np.uint8)
            c[:len(a)] = a
            a = c
        res = cv2.matchTemplate(a, b, mode) # cv2.TM_CCORR_NORMED
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #cv2.imshow("templatematch", res)
        print(min_val, f">>{max_val}<<", min_loc, max_loc)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if mode == cv2.TM_SQDIFF_NORMED:
            return min_val
        return max_val


    def __decide_corr(self, c):
        return c < self.correlation_value


    def analyze_bbox(self, croped, identifier):
        pn = self.posinv_name(identifier)
        if self.mode == Algorithm.MODE_FIND_CELLS:
            if pn not in self.cell_templates:
                # TODO if size is original, store and continue
                #if len(croped) == bh and len(croped[0]) == self.b.bw:
                self.cell_templates[pn] = croped
                still_missing = len(self.get_missing_cells())
                print(f"{pn} added, {still_missing} still missing...")

                """f = f"template_images"
                c = pn.replace(" at","").replace(" ", "_").replace(",", "").replace("\"", "").replace("(","").replace(")","").replace("CellReference","")
                rf = os.path.join(self.output_folder, f)
                if not os.path.exists(rf):
                    os.makedirs(rf)
                cv2.imwrite(os.path.join(rf, c + ".png"), croped)"""

                if still_missing == 0:
                    return True, {} # abort for the first run when all templates are collected
            return False, None
        
        # correlate with existing template
        print(f"correlate {identifier}...")
        corr = self.template(croped, self.cell_templates[pn])
        if self.__decide_corr(corr):
            return True, {"First Template": self.cell_templates[pn][0]}

        return False, None