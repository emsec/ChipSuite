#!/usr/bin/env python3
import cv2
import numpy as np
import math

from chipsuite.algorithm import Algorithm
from chipsuite.algorithm3 import Algorithm3
from chipsuite import algorithm
from chipsuite import cvhelper

DEBUG = False

class Algorithm4(Algorithm):
    def __init__(self, *args, **kwargs):
        self.max_features = 500
        self.keep_percent = 0.1
        self.max_rotation = 13
        self.correlation_value = 0.9
        super().__init__(*args, **kwargs)


#with open("65nm/alg4.pickle", "wb") as alg4_pickle:
#    pickle.dump((algorithm.template_images, algorithm.mean_images, algorithm.cell_count), alg4_pickle)
#with open("65nm/alg4.pickle", "rb") as alg4_pickle:
#    template_images, mean_images, cell_count = pickle.load(alg4_pickle)
#for key, mean_image in algorithm.mean_images.items():
#    algorithm.mean_results[key] = (mean_image / algorithm.cell_count[key]).astype("uint8")
    def prepare(self):
        self.template_images = {}
        self.mean_images = {}
        self.mean_results = {}
        self.cell_count = {}
        print("FIRST RUN")
        self.analyze_loop(mode=Algorithm.MODE_ONLY_CELLS)
        for key, mean_image in self.mean_images.items():
            self.mean_results[key] = (mean_image / self.cell_count[key]).astype("uint8")
        print("FIRST RUN DONE")

    def warp_onto_template(self, template, croped):
        # https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/
        orb = cv2.ORB_create(self.max_features)
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(croped, None)
        if not kp1 or not kp2:
            return [], None

        #flann = cv2.FlannBasedMatcher({
        #    "algorithm": 1, # FLANN_INDEX_KDTREE
        #    "trees": 5,
        #    }, {
        #    "checks": 50,
        #    })
        #matches = flann.knnMatch(des1, des2, k=2)

        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(des1, des2, None)
        # quality of matches?

        matches = sorted(matches, key = lambda x: x.distance)
        keep = int(len(matches) * self.keep_percent)
        matches = matches[:keep]
        if DEBUG: print("matches:", ",".join(str(m.distance) for m in matches))
        pts1 = np.zeros((len(matches), 2), dtype="float")
        pts2 = np.zeros((len(matches), 2), dtype="float")
        for (i, m) in enumerate(matches):
            pts1[i] = kp1[m.queryIdx].pt
            pts2[i] = kp2[m.trainIdx].pt
        H = cv2.findHomography(pts2, pts1, method = cv2.RANSAC)[0]
        if DEBUG: print("H:", H)
        if H is None:
            return [], None
        h, w = template.shape[:2]
        aligned = cv2.warpPerspective(croped, H, (w, h))
        return aligned, H

    def analyze_bbox(self, croped, identifier):
        pn = Algorithm3.posinv_name(identifier)
        if self.mode == Algorithm.MODE_ONLY_CELLS:
            if DEBUG: print(f"== {pn} ==")
            if pn not in self.template_images:
                # put current bbox in the mean images dict
                self.template_images[pn] = croped
                self.mean_images[pn] = croped.astype(int)
                self.cell_count[pn] = 1
                if DEBUG: print("new template...")
            else:
                template = self.template_images[pn]
                if DEBUG:
                    cvhelper.imshowx("croped", croped, algorithm.CV_X + 1000, 0)
                    cvhelper.imshowx("template", template, algorithm.CV_X + 500, 0)

                """(bh, bw) = croped.shape[:2]
                bw += self.cx1 + self.cx2
                bh += self.cy1 + self.cy2
                bx = -self.cx1
                by = -self.cy1
                if bw > 0 and bh > 0:
                    croped = croped[max(0,by):max(0,by+bh), max(0,bx):max(0,bx+bw)].copy()
                else:
                    return False, None"""

                aligned, H = self.warp_onto_template(template, croped)
                if H is None:
                    if self.cell_count[pn] == 1:
                        # also reset to the new image
                        self.template_images[pn] = croped
                        self.mean_images[pn] = croped.astype(int)
                    return False, None

                if DEBUG:
                    overlay = template.copy()
                    output = aligned.copy()
                    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
                    cvhelper.imshowx("overlay", output, algorithm.CV_X, 0)
                    cvhelper.imshowx("RG overlay", cv2.merge([np.zeros(template.shape[:2], dtype="uint8"), aligned, template]), algorithm.CV_X, 400)



                rot = cv2.decomposeHomographyMat(H, np.identity(3))[1]
                if any(not Algorithm4.isRotationMatrix(mat) for mat in rot):
                    if DEBUG: print("invalid matrix, skip")
                    if self.cell_count[pn] == 1:
                        # also reset to the new image
                        self.template_images[pn] = croped
                        self.mean_images[pn] = croped.astype(int)
                    return False, None
                for mat in rot:
                    if DEBUG: print(Algorithm4.rotationMatrixToEulerAngles(mat)/math.pi*180)
                rot = min(max(np.abs(Algorithm4.rotationMatrixToEulerAngles(mat))) for mat in rot) / math.pi * 180
                if DEBUG: print("maximum rotation:", rot)
                if rot > self.max_rotation:
                    if self.cell_count[pn] == 1:
                        # also reset to the new image
                        self.template_images[pn] = croped
                        self.mean_images[pn] = croped.astype(int)
                    return False, None

                self.mean_images[pn] += aligned
                self.cell_count[pn] += 1



                if DEBUG:
                    key = cv2.waitKey()
                    cv2.destroyAllWindows()
                    if key == ord("q"):
                        self.matrix = H
                        return True, {"Output": output}
            return False, None

        # correlate with existing template only
        if pn not in self.mean_results:
            return False, None

        print(f"correlate {identifier}...")
        #aligned = self.warp_onto_template(self.mean_results[pn], croped)
        corr = Algorithm3.template(croped, self.mean_results[pn])
        if corr < self.correlation_value:
            return "FILL" not in identifier, {"Mean Result": self.mean_results[pn]}

        return False, None

    # from https://learnopencv.com/rotation-matrix-to-euler-angles/
    @staticmethod
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    @staticmethod
    def rotationMatrixToEulerAngles(R):
        assert(Algorithm4.isRotationMatrix(R))
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])